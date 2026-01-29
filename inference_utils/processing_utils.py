# %%
# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

# Change this path whenever you have a new sample
FILE_ID = "13_1"  
IMAGE_PATH = f"/home/fetalusr1/Fetal-Head-Segmentation-master/IMG_20250329_13_1.nii"

# This automatically names your outputs based on the input
MASK_OUTPUT_PATH = f"./FilteredRes/segmentation_result_{FILE_ID}_interpolated.nii.gz"
REPORT_ZIP_NAME = f"3D_Full_Report_{FILE_ID}.zip"

print(f"🚀 Ready to process: {IMAGE_PATH}")
print(f"💾 Results will save to: {MASK_OUTPUT_PATH}")

# %%
import os
# optimizing memory allocation to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print("✅ Memory fragmentation rules applied.")

# %%
import os
import torch

# 1. Help PyTorch manage fragmented memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# 2. Clear any lingering cache
torch.cuda.empty_cache()

print(f"✅ Memory settings applied. Free memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# %%
import os
import sys

# Get the path to your current environment
conda_prefix = sys.prefix
lib_path = os.path.join(conda_prefix, 'lib')

# Force this path to the front of the line
os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

print(f"✅ Forced Library Path: {lib_path}")

# %%
import torch

try:
    # Try a simple calculation on the GPU
    x = torch.tensor([1.0, 2.0]).cuda()
    y = torch.tensor([3.0, 4.0]).cuda()
    z = x * y
    print("--------------------------------------------------")
    print(f"🎉 SUCCESS: GPU Math works! Result: {z.cpu().numpy()}")
    print("--------------------------------------------------")
except RuntimeError as e:
    print("--------------------------------------------------")
    print(f"❌ FAILURE: {e}")
    print("--------------------------------------------------")

# %%
from PIL import Image
import torch
import numpy as np
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
import matplotlib.pyplot as plt
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
from inference_utils.processing_utils import process_intensity_image
from inference_utils.processing_utils import read_nifti
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from skimage.measure import regionprops, label
from skimage.transform import resize


out_probs = []
predicted_masks = []

# %% [markdown]
# ## Loading the Finetuned BiomedParse model

# %%
# Build model config
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
finetuned_pth = '/home/fetalusr1/Fetal-Head-Segmentation-master/model_state_dict.pt' # Replace with the path to your finetuned checkpoint

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained=finetuned_pth).eval().cuda()

with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

# %% [markdown]
# ## Utilities

# %%
def get_segmentation_masks(original_image, segmentation_masks, texts, rotate=0):
    ''' Plot a list of segmentation mask over an image showing only the segmented region.
    '''
    original_image = original_image[:, :, :3]

    segmented_images = []

    for i, mask in enumerate(segmentation_masks):
        segmented_image = original_image.copy()
        segmented_image[mask <= 0.5] = [0, 0, 0]
        segmented_images.append(segmented_image)
        
    return segmented_images

# %%
def inference_nifti(file_path, text_prompts, is_CT, slice_idx, site=None, HW_index=(0, 1), channel_idx=None, rotate=0):

    image = read_nifti(file_path, is_CT, slice_idx, site=site, HW_index=HW_index, channel_idx=channel_idx)
    
    pred_mask,out_prob = interactive_infer_image(model, Image.fromarray(image), text_prompts)
    predicted_masks.append(pred_mask)
    segmented_images = get_segmentation_masks(image, pred_mask, text_prompts, rotate=rotate)
    out_probs.append(out_prob)
    
    return image, pred_mask, segmented_images

# %% [markdown]
# ### Post-processing Utility

# %%
def process_predicted_volume(volume_data, threshold_factor=0.35, output_prefix='processed'):
    """
    Process the predicted volume to filter based on ellipse measurements.
    """
    data = volume_data
    print(f"Processing volume with shape: {data.shape}")
    
    # Calculate measurements for all slices
    results = []
    z_0 = data.shape[2] // 2  # Reference slice (middle slice)
    
    print(f"Reference slice: {z_0}")
    
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]
        
        # Skip empty slices
        if np.sum(slice_data) == 0:
            continue
            
        # Binarize the slice
        slice_bin = np.where(slice_data > 0, 1, 0).astype(np.uint8)
        
        # Fill holes
        slice_bin_filled = sitk.BinaryFillhole(sitk.GetImageFromArray(slice_bin))
        slice_bin_filled = sitk.GetArrayFromImage(slice_bin_filled)
        
        # Get region properties
        labeled_image = label(slice_bin_filled)
        props = regionprops(labeled_image)
        
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
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    print(f"Found {len(results)} regions across {len(df_results['slice_index'].unique())} slices")
    
    # Get reference slice measurements for filtering
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
    
    # Define thresholds
    major_axis_length_threshold = major_axis_length_std * (1 - threshold_factor)
    minor_axis_length_threshold = minor_axis_length_std * (1 - threshold_factor)
    
    print(f"Reference measurements - Major: {major_axis_length_std:.2f}, Minor: {minor_axis_length_std:.2f}")
    print(f"Filtering thresholds - Major: {major_axis_length_threshold:.2f}, Minor: {minor_axis_length_threshold:.2f}")
    
    # Filter based on thresholds
    filtered_df = df_results[
        (df_results['major_axis_length'] >= major_axis_length_threshold) &
        (df_results['minor_axis_length'] >= minor_axis_length_threshold)
    ]
    
    print(f"After filtering: {len(filtered_df)} regions in {len(filtered_df['slice_index'].unique())} slices")
    
    # In filtered_df, in case of repeated slices, keep the one with maximum major axis length
    filtered_df = filtered_df.loc[filtered_df.groupby('slice_index')['major_axis_length'].idxmax()]
    
    # Create filtered volume
    filtered_slices = filtered_df['slice_index'].unique()
    filtered_volume = np.zeros_like(data)
    
    for slice_idx in range(data.shape[2]):
        if slice_idx in filtered_slices:
            filtered_volume[:, :, slice_idx] = data[:, :, slice_idx]
    
    return filtered_volume, filtered_df

# %% [markdown]
# ### Interpolation Utility

# %%
def interpolate_blank_slices(image_path, processed_volume, blank_slices, predicted_masks, delta=1):
    """
    Interpolate blank slices in the processed volume using the previous slice.
    """
    vol_data = nib.load(image_path).get_fdata()
    central_slice = vol_data.shape[2] // 2
    
    for slice_idx in blank_slices:
        # Ensure we have a valid previous slice
        prev_slice_idx = slice_idx - delta
        if prev_slice_idx < 0 or prev_slice_idx >= len(predicted_masks):
            continue
            
        # Get the previous mask
        prev_mask = predicted_masks[prev_slice_idx][0]  # Get first mask from the list
        
        #update predicted_masks
        predicted_masks[slice_idx] = [prev_mask.copy()]  # Store the previous mask
        # Ensure the previous mask is not empty
        if np.sum(prev_mask) == 0:
            print(f"Warning: Previous mask for slice {prev_slice_idx} is empty. Skipping interpolation for slice {slice_idx}.")
            continue
        # Scale the mask based on position relative to center
        if slice_idx < central_slice: 
            # Increase the mask size by 0.5%
            new_mask = prev_mask * 1.005
        else:
            # Decrease the mask size by 0.5%
            new_mask = prev_mask * 0.995
        
        # Read the original image for this slice
        image = read_nifti(image_path, is_CT=False, slice_idx=slice_idx, site=None, HW_index=(0, 1), channel_idx=None)
        
        # Get the segmented image
        new_segmented_image = get_segmentation_masks(image, [new_mask], ['fetal head'], rotate=0)[0]
        
        # Convert RGB segmentation to grayscale if needed
        if len(new_segmented_image.shape) == 3:
            gray_mask = np.mean(new_segmented_image, axis=2)
        else:
            gray_mask = new_segmented_image
        
        # Resize to match volume dimensions and store
        from skimage.transform import resize
        processed_volume[:, :, slice_idx] = resize(gray_mask, (vol_data.shape[0], vol_data.shape[1]), preserve_range=True)
    
    return processed_volume

# %% [markdown]
# ## Working

# %%
image_path = '/home/fetalusr1/Fetal-Head-Segmentation-master/IMG_20250329_13_1.nii'
text_prompt = ['fetal head']
vol = nib.load(image_path)
vol_data = vol.get_fdata()
vol_data.shape

# %%
# Initialize volume to store all prediction masks
pred_volume = np.zeros((vol_data.shape[0], vol_data.shape[1], vol_data.shape[2]))

counter = 0
for slice_idx in range(vol_data.shape[2]):
    image, pred_mask, segmentation_mask = inference_nifti(image_path, text_prompt, is_CT=False, slice_idx=slice_idx, site=None, rotate=0)
    
    # Convert RGB segmentation mask to grayscale
    if len(segmentation_mask[0].shape) == 3:
        # Convert to grayscale by taking the mean across color channels
        gray_mask = np.mean(segmentation_mask[0], axis=2)
    else:
        gray_mask = segmentation_mask[0]
    
    # Store the prediction mask in the volume
    pred_volume[:, :, slice_idx] = resize(gray_mask, (vol_data.shape[0], vol_data.shape[1]), preserve_range=True)

# Post processing

processed_volume, filtered_measurements = process_predicted_volume(
    pred_volume, 
    threshold_factor=0.4,  # Adjust as needed
    output_prefix='3_2'
)

print(f"Original volume had {np.sum(pred_volume > 0)} non-zero voxels")
print(f"Processed volume has {np.sum(processed_volume > 0)} non-zero voxels")

# %%
#Get the first slice that survived filtering
first_filtered_slice = min(filtered_measurements['slice_index'].unique())
last_filtered_slice = max(filtered_measurements['slice_index'].unique())
print(f"First filtered slice: {first_filtered_slice}")
print(f"Last filtered slice: {last_filtered_slice}")
#from the filtered slice to the center slice, get all the slices which are blank
blank_slices = []
for slice_idx in range(first_filtered_slice, last_filtered_slice + 1):
    if np.sum(processed_volume[:, :, slice_idx]) == 0:
        blank_slices.append(slice_idx)
# Print the blank slices
print(f"Blank slices from {first_filtered_slice} to {vol_data.shape[2]-1}: {blank_slices}")

# %%
import os

# Create results directories if they don't exist
os.makedirs('./results', exist_ok=True)
os.makedirs('./FilteredRes', exist_ok=True)

# Load original NIfTI for header info
original_nii = nib.load(image_path)

# Save raw prediction
pred_nii = nib.Nifti1Image(pred_volume, original_nii.affine, original_nii.header)
raw_filename = f'./results/segmentation_RAW.nii.gz'
nib.save(pred_nii, raw_filename)
print(f"Raw prediction saved to {raw_filename}")

# Save processed prediction
processed_nii = nib.Nifti1Image(processed_volume, original_nii.affine, original_nii.header)
processed_filename = f'./FilteredRes/segmentation_fil.nii.gz'
nib.save(processed_nii, processed_filename)
print(f"Processed prediction saved to {processed_filename}")

interpolated_volume = interpolate_blank_slices(image_path, processed_volume, blank_slices, predicted_masks, delta=1)
# Save interpolated prediction
interpolated_nii = nib.Nifti1Image(interpolated_volume, original_nii.affine, original_nii.header)
interpolated_filename = f'./FilteredRes/segmentation_inter.nii.gz'
nib.save(interpolated_nii, interpolated_filename)
print(f"Interpolated prediction saved to {interpolated_filename}")

# %%
import nibabel as nib
import numpy as np

# 1. Check the physical data shape
print(f"📊 Volume Array Shape: {processed_volume.shape}")
print(f"📊 Volume Data Type: {processed_volume.dtype}")
print(f"📊 Max Value in Volume: {np.max(processed_volume)}")

# 2. Check the "Ruler" (Header)
orig_nii = nib.load(image_path)
print(f"📏 Original Zooms (Spacing): {orig_nii.header.get_zooms()}")
print(f"🗺️ Original Affine Matrix:\n{orig_nii.affine}")

# 3. Check for the "Veto" impact
active_slices = np.where(np.sum(processed_volume, axis=(0, 1)) > 0)[0]
print(f"🧩 Number of slices containing data: {len(active_slices)}")

# %%
'''import scipy.ndimage as ndimage

# 1. Expand the mask slightly (Dilation)
# This adds a 2-3 pixel buffer around the head so we don't cut into the skull
print("🛡️ Expanding mask buffer...")
structure = ndimage.generate_binary_structure(3, 1)
# dilate by 2 iterations to ensure outer skull is included
expanded_mask = ndimage.binary_dilation(interpolated_volume > 0, structure=structure, iterations=2)

# 2. Smooth the expanded mask (to keep it natural, not jagged)
expanded_mask = ndimage.median_filter(expanded_mask.astype(np.float32), size=3)

# 3. Multiply by Original Intensity
# Now we use the expanded mask so we don't strip too much
final_output_vol = vol_data * expanded_mask

# 4. Save with Header Correction (Fixing the Squished look)
original_zooms = original_nii.header.get_zooms()
improved_nii = nib.Nifti1Image(final_output_vol, original_nii.affine, original_nii.header)
improved_nii.header.set_zooms(original_zooms)

improved_filename = f'./FilteredRes/{FILE_ID}_PRESERVED.nii.gz'
nib.save(improved_nii, improved_filename)
print(f"✅ Anatomically Preserved Result Saved: {improved_filename}")'''

# %%
original_nii.header

# %%
import scipy.ndimage as ndimage
import nibabel as nib
import numpy as np

# 1. CREATE MASK FROM EXISTING WORKING VOLUME
# We start with 'interpolated_volume' which you confirmed is correct/working in memory.
# Any pixel with data becomes part of the mask.
current_mask = interpolated_volume > 0

# 2. CONSERVATIVE DILATION (The Skull Fix)
# Iterations=2: Expands the mask just enough to touch the inner skull without overshooting.
print("🛡️ Applying conservative dilation (2 iterations)...")
structure = ndimage.generate_binary_structure(3, 1)
dilated_mask = ndimage.binary_dilation(current_mask, structure=structure, iterations=3)

# Optional: Slight smoothing to keep edges organic
dilated_mask = ndimage.median_filter(dilated_mask.astype(np.float32), size=3)

# 3. APPLY TO ORIGINAL TEXTURE
# Use the dilated mask to grab the brain + skull edge from the original data
final_output = vol_data * dilated_mask

# 4. SAVE USING NOTEBOOK LOGIC (No 'set_zooms' override)
# We use the exact header/affine from 'original_nii' just like your working Cell 96.
final_nii = nib.Nifti1Image(final_output.astype(np.float32), original_nii.affine, original_nii.header)

save_path = f'./FilteredRes/{FILE_ID}_Corrected.nii.gz'
nib.save(final_nii, save_path)
print(f"✅ Success. Saved corrected output to: {save_path}")



