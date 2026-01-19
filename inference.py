"""
BiomedParse Medical Image Segmentation Pipeline
==============================================

This script performs automated segmentation of medical images (NIfTI format) using 
the BiomedParse model. The pipeline includes:
1. Raw prediction generation
2. Volume filtering based on morphological properties
3. Interpolation of missing slices
4. Output generation in multiple formats

Author: BiomedParse Team
"""

# Standard library imports
import os
import argparse
from pathlib import Path

# Third-party imports
from PIL import Image
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from skimage.measure import regionprops, label
from skimage.transform import resize
import matplotlib.pyplot as plt

# Local imports
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
from inference_utils.processing_utils import process_intensity_image, read_nifti

# Global variables for storing predictions
out_probs = []
predicted_masks = []

def initialize_model():
    """
    Initialize the BiomedParse model with pretrained weights.
    
    Returns:
        model: Loaded and configured BiomedParse model
    """
    print("Initializing BiomedParse model...")
    
    # Build model configuration
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    finetuned_pth = 'output/biomed_seg_lang_v1.yaml_conf~/run_1/00003735/default/model_state_dict.pt' # Replace with the path to your finetuned checkpoint
    
    if not os.path.exists(finetuned_pth):
        raise FileNotFoundError(f"Model weights not found at {finetuned_pth}")
    
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained=finetuned_pth).eval().cuda()

    # Initialize text embeddings for biomedical classes
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    
    print("Model initialized successfully!")
    return model

def get_segmentation_masks(original_image, segmentation_masks, texts, rotate=0):
    """
    Apply segmentation masks to the original image, showing only segmented regions.
    
    Args:
        original_image (np.ndarray): Original input image
        segmentation_masks (list): List of binary masks
        texts (list): Text prompts used for segmentation
        rotate (int): Rotation parameter (unused in current implementation)
    
    Returns:
        list: List of segmented images with masks applied
    """
    # Ensure image has RGB channels
    original_image = original_image[:, :, :3]
    segmented_images = []

    for i, mask in enumerate(segmentation_masks):
        segmented_image = original_image.copy()
        # Set background pixels to black where mask is weak
        segmented_image[mask <= 0.5] = [0, 0, 0]
        segmented_images.append(segmented_image)
        
    return segmented_images

def inference_nifti(model, file_path, text_prompts, is_CT, slice_idx, site=None, HW_index=(0, 1), channel_idx=None, rotate=0):
    """
    Perform inference on a single slice of a NIfTI volume.
    
    Args:
        model: BiomedParse model instance
        file_path (str): Path to NIfTI file
        text_prompts (list): Text descriptions for segmentation targets
        is_CT (bool): Whether the image is CT scan
        slice_idx (int): Index of the slice to process
        site (str, optional): Anatomical site information
        HW_index (tuple): Height/Width dimension indices
        channel_idx (int, optional): Channel index for multi-channel images
        rotate (int): Rotation parameter
    
    Returns:
        tuple: (original_image, prediction_mask, segmented_images)
    """
    # Read the NIfTI slice
    image = read_nifti(file_path, is_CT, slice_idx, site=site, HW_index=HW_index, channel_idx=channel_idx)
    
    # Perform inference using the model
    pred_mask, out_prob = interactive_infer_image(model, Image.fromarray(image), text_prompts)
    
    # Store results globally for later use
    predicted_masks.append(pred_mask)
    out_probs.append(out_prob)
    
    # Generate segmented visualizations
    segmented_images = get_segmentation_masks(image, pred_mask, text_prompts, rotate=rotate)
    
    return image, pred_mask, segmented_images

def process_predicted_volume(volume_data, threshold_factor=0.35, output_prefix='processed'):
    """
    Filter the predicted volume based on morphological properties of segmented regions.
    
    This function analyzes each slice for region properties (major/minor axis lengths,
    centroids, etc.) and filters out slices that don't meet size thresholds based on
    a reference slice (typically the middle slice).
    
    Args:
        volume_data (np.ndarray): 3D volume with prediction masks
        threshold_factor (float): Factor for filtering threshold (0.0-1.0)
        output_prefix (str): Prefix for output files
    
    Returns:
        tuple: (filtered_volume, filtered_measurements_dataframe)
    """
    data = volume_data
    print(f"Processing volume with shape: {data.shape}")
    
    # Calculate morphological measurements for all slices
    results = []
    z_0 = data.shape[2] // 2  # Reference slice (middle slice)
    print(f"Reference slice: {z_0}")
    
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]
        
        # Skip empty slices
        if np.sum(slice_data) == 0:
            continue
            
        # Binarize the slice data
        slice_bin = np.where(slice_data > 0, 1, 0).astype(np.uint8)
        
        # Fill holes in the binary mask using SimpleITK
        slice_bin_filled = sitk.BinaryFillhole(sitk.GetImageFromArray(slice_bin))
        slice_bin_filled = sitk.GetArrayFromImage(slice_bin_filled)
        
        # Extract region properties using scikit-image
        labeled_image = label(slice_bin_filled)
        props = regionprops(labeled_image)
        
        # Store measurements for each connected component
        for prop in props:
            results.append({
                'slice_index': i,
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length,
                'centroid_x': prop.centroid[1],
                'centroid_y': prop.centroid[0],
                'orientation': prop.orientation,
                'area': prop.area
            })
    
    # Create DataFrame for analysis
    df_results = pd.DataFrame(results)
    print(f"Found {len(results)} regions across {len(df_results['slice_index'].unique())} slices")
    
    # Get reference measurements from the middle slice
    standard_slice_data = df_results[df_results['slice_index'] == z_0]
    
    if standard_slice_data.empty:
        print(f"Warning: No data found in reference slice {z_0}")
        # Use overall median as fallback
        major_axis_length_std = df_results['major_axis_length'].median()
        minor_axis_length_std = df_results['minor_axis_length'].median()
        centroid_x_std = df_results['centroid_x'].median()
        centroid_y_std = df_results['centroid_y'].median()
    else:
        major_axis_length_std = standard_slice_data['major_axis_length'].values[0]
        minor_axis_length_std = standard_slice_data['minor_axis_length'].values[0]
        centroid_x_std = standard_slice_data['centroid_x'].values[0]
        centroid_y_std = standard_slice_data['centroid_y'].values[0]
    
    # Define filtering thresholds based on reference measurements
    major_axis_length_threshold = major_axis_length_std * (1 - threshold_factor)
    minor_axis_length_threshold = minor_axis_length_std * (1 - threshold_factor)
    
    print(f"Reference measurements - Major: {major_axis_length_std:.2f}, Minor: {minor_axis_length_std:.2f}")
    print(f"Filtering thresholds - Major: {major_axis_length_threshold:.2f}, Minor: {minor_axis_length_threshold:.2f}")
    
    # Apply filtering based on size thresholds
    filtered_df = df_results[
        (df_results['major_axis_length'] >= major_axis_length_threshold) &
        (df_results['minor_axis_length'] >= minor_axis_length_threshold)
    ]
    
    print(f"After filtering: {len(filtered_df)} regions in {len(filtered_df['slice_index'].unique())} slices")
    
    # In case of multiple regions per slice, keep the one with maximum major axis length
    filtered_df = filtered_df.loc[filtered_df.groupby('slice_index')['major_axis_length'].idxmax()]
    
    # Create filtered volume based on surviving slices
    filtered_slices = filtered_df['slice_index'].unique()
    filtered_volume = np.zeros_like(data)
    
    for slice_idx in range(data.shape[2]):
        if slice_idx in filtered_slices:
            filtered_volume[:, :, slice_idx] = data[:, :, slice_idx]
    
    return filtered_volume, filtered_df

def interpolate_blank_slices(image_path, processed_volume, blank_slices, predicted_masks, delta=1):
    """
    Interpolate missing slices in the processed volume using adjacent slice information.
    
    This function fills in blank slices by copying and slightly scaling masks from
    previous slices, with scaling direction depending on position relative to center.
    
    Args:
        image_path (str): Path to original NIfTI file
        processed_volume (np.ndarray): Volume after filtering
        blank_slices (list): Indices of slices that need interpolation
        predicted_masks (list): List of all predicted masks
        delta (int): Step size for looking back to previous slice
    
    Returns:
        np.ndarray: Volume with interpolated slices filled
    """
    vol_data = nib.load(image_path).get_fdata()
    central_slice = vol_data.shape[2] // 2
    
    print(f"Interpolating {len(blank_slices)} blank slices...")
    
    for slice_idx in blank_slices:
        # Find a valid previous slice
        prev_slice_idx = slice_idx - delta
        if prev_slice_idx < 0 or prev_slice_idx >= len(predicted_masks):
            continue
            
        # Get the previous mask
        prev_mask = predicted_masks[prev_slice_idx][0]  # Get first mask from the list
        
        # Update predicted_masks list with interpolated mask
        predicted_masks[slice_idx] = [prev_mask.copy()]
        
        # Ensure the previous mask is not empty
        if np.sum(prev_mask) == 0:
            print(f"Warning: Previous mask for slice {prev_slice_idx} is empty. Skipping interpolation for slice {slice_idx}.")
            continue
            
        # Scale the mask based on position relative to center
        # This accounts for natural size variation across the volume
        if slice_idx < central_slice: 
            # Slightly increase mask size for slices before center
            new_mask = prev_mask * 1.005
        else:
            # Slightly decrease mask size for slices after center
            new_mask = prev_mask * 0.995
        
        # Read the original image for this slice
        image = read_nifti(image_path, is_CT=False, slice_idx=slice_idx, site=None, HW_index=(0, 1), channel_idx=None)
        
        # Generate segmented image with the scaled mask
        new_segmented_image = get_segmentation_masks(image, [new_mask], ['fetal head'], rotate=0)[0]
        
        # Convert RGB segmentation to grayscale if needed
        if len(new_segmented_image.shape) == 3:
            gray_mask = np.mean(new_segmented_image, axis=2)
        else:
            gray_mask = new_segmented_image
        
        # Resize to match volume dimensions and store
        processed_volume[:, :, slice_idx] = resize(gray_mask, (vol_data.shape[0], vol_data.shape[1]), preserve_range=True)
    
    return processed_volume

def save_results(pred_volume, processed_volume, interpolated_volume, original_nii, output_dir, file_id):
    """
    Save all segmentation results to NIfTI files.
    
    Args:
        pred_volume (np.ndarray): Raw prediction volume
        processed_volume (np.ndarray): Filtered prediction volume
        interpolated_volume (np.ndarray): Interpolated prediction volume
        original_nii: Original NIfTI image (for header/affine info)
        output_dir (str): Output directory path
        file_id (str): Identifier for output files
    
    Returns:
        dict: Dictionary containing paths to saved files
    """
    # Create output directories
    results_dir = os.path.join(output_dir, 'results')
    filtered_dir = os.path.join(output_dir, 'FilteredRes')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)
    
    saved_files = {}
    
    # Save raw prediction
    pred_nii = nib.Nifti1Image(pred_volume, original_nii.affine, original_nii.header)
    raw_filename = os.path.join(results_dir, f'segmentation_result_{file_id}_raw.nii.gz')
    nib.save(pred_nii, raw_filename)
    saved_files['raw'] = raw_filename
    print(f"Raw prediction saved to {raw_filename}")

    # Save processed prediction
    processed_nii = nib.Nifti1Image(processed_volume, original_nii.affine, original_nii.header)
    processed_filename = os.path.join(filtered_dir, f'segmentation_result_{file_id}_filtered.nii.gz')
    nib.save(processed_nii, processed_filename)
    saved_files['filtered'] = processed_filename
    print(f"Processed prediction saved to {processed_filename}")

    # Save interpolated prediction
    interpolated_nii = nib.Nifti1Image(interpolated_volume, original_nii.affine, original_nii.header)
    interpolated_filename = os.path.join(filtered_dir, f'segmentation_result_{file_id}_interpolated.nii.gz')
    nib.save(interpolated_nii, interpolated_filename)
    saved_files['interpolated'] = interpolated_filename
    print(f"Interpolated prediction saved to {interpolated_filename}")
    
    return saved_files

def run_segmentation_pipeline(image_path, text_prompt=['fetal head'], threshold_factor=0.4, output_dir='./'):
    """
    Main segmentation pipeline that orchestrates the entire process.
    
    Args:
        image_path (str): Path to input NIfTI file
        text_prompt (list): Text descriptions for segmentation targets
        threshold_factor (float): Filtering threshold factor
        output_dir (str): Output directory for results
    
    Returns:
        dict: Dictionary containing paths to all saved results
    """
    # Clear global variables
    global out_probs, predicted_masks
    out_probs = []
    predicted_masks = []
    
    # Validate input file
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input file not found: {image_path}")
    
    print(f"Starting segmentation pipeline for: {image_path}")
    print(f"Text prompt: {text_prompt}")
    
    # Initialize model
    model = initialize_model()
    
    # Load NIfTI volume
    vol = nib.load(image_path)
    vol_data = vol.get_fdata()
    print(f"Loaded volume with shape: {vol_data.shape}")
    
    # Initialize prediction volume
    pred_volume = np.zeros((vol_data.shape[0], vol_data.shape[1], vol_data.shape[2]))
    
    # Process each slice
    print("Running inference on all slices...")
    for slice_idx in range(vol_data.shape[2]):
        if slice_idx % 10 == 0:  # Progress indicator
            print(f"Processing slice {slice_idx}/{vol_data.shape[2]}")
            
        image, pred_mask, segmentation_mask = inference_nifti(
            model, image_path, text_prompt, is_CT=False, slice_idx=slice_idx, 
            site=None, rotate=3
        )
        
        # Convert RGB segmentation mask to grayscale
        if len(segmentation_mask[0].shape) == 3:
            gray_mask = np.mean(segmentation_mask[0], axis=2)
        else:
            gray_mask = segmentation_mask[0]
        
        # Store the prediction mask in the volume
        pred_volume[:, :, slice_idx] = resize(
            gray_mask, (vol_data.shape[0], vol_data.shape[1]), preserve_range=True
        )
    
    print("Inference completed!")
    
    # Post-processing: Filter volume based on morphological properties
    print("Applying morphological filtering...")
    processed_volume, filtered_measurements = process_predicted_volume(
        pred_volume, 
        threshold_factor=threshold_factor,
        output_prefix=Path(image_path).stem
    )
    
    print(f"Original volume had {np.sum(pred_volume > 0)} non-zero voxels")
    print(f"Processed volume has {np.sum(processed_volume > 0)} non-zero voxels")
    
    # Identify blank slices that need interpolation
    if not filtered_measurements.empty:
        first_filtered_slice = min(filtered_measurements['slice_index'].unique())
        last_filtered_slice = max(filtered_measurements['slice_index'].unique())
        print(f"First filtered slice: {first_filtered_slice}")
        print(f"Last filtered slice: {last_filtered_slice}")
        
        # Find blank slices in the filtered range
        blank_slices = []
        for slice_idx in range(first_filtered_slice, last_filtered_slice + 1):
            if np.sum(processed_volume[:, :, slice_idx]) == 0:
                blank_slices.append(slice_idx)
        
        print(f"Blank slices requiring interpolation: {blank_slices}")
        
        # Interpolate blank slices
        if blank_slices:
            print("Performing slice interpolation...")
            interpolated_volume = interpolate_blank_slices(
                image_path, processed_volume, blank_slices, predicted_masks, delta=1
            )
        else:
            interpolated_volume = processed_volume.copy()
            print("No interpolation needed.")
    else:
        print("Warning: No slices survived filtering. Skipping interpolation.")
        interpolated_volume = processed_volume.copy()
    
    # Save all results
    file_id = Path(image_path).stem
    saved_files = save_results(
        pred_volume, processed_volume, interpolated_volume, 
        vol, output_dir, file_id
    )
    
    return saved_files

def get_user_input():
    """
    Get input file path from user with validation.
    
    Returns:
        str: Validated path to NIfTI file
    """
    while True:
        image_path = input("Enter the path to your NIfTI file (.nii or .nii.gz): ").strip()
        
        # Remove quotes if present
        image_path = image_path.strip('"\'')
        
        if os.path.exists(image_path):
            if image_path.lower().endswith(('.nii', '.nii.gz')):
                return image_path
            else:
                print("Error: File must be a NIfTI file (.nii or .nii.gz)")
        else:
            print(f"Error: File not found: {image_path}")
            print("Please check the path and try again.")

def main():
    """
    Main function to run the segmentation pipeline with user interaction.
    """
    print("=" * 60)
    print("BiomedParse Medical Image Segmentation Pipeline")
    print("=" * 60)
    print()
    
    # Get input from user
    image_path = get_user_input()
    
    # Optional: Get text prompt from user
    print("\nDefault segmentation target: 'fetal head'")
    custom_prompt = input("Enter custom segmentation target (or press Enter for default): ").strip()
    
    if custom_prompt:
        text_prompt = [custom_prompt]
    else:
        text_prompt = ['fetal head']
    
    # Optional: Get output directory
    output_dir = input("Enter output directory (or press Enter for current directory): ").strip()
    if not output_dir:
        output_dir = './'
    
    try:
        # Run the segmentation pipeline
        print("\n" + "=" * 60)
        print("Starting segmentation pipeline...")
        print("=" * 60)
        
        saved_files = run_segmentation_pipeline(
            image_path=image_path,
            text_prompt=text_prompt,
            threshold_factor=0.4,
            output_dir=output_dir
        )
        
        print("\n" + "=" * 60)
        print("SEGMENTATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type.capitalize()}: {file_path}")
        
        print(f"\nAll results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError during segmentation: {str(e)}")
        print("Please check your input file and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


