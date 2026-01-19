# Segmentation and Reconstruction of the Fetal Head from Fetal Ultrasound Volumes

## Pipeline Overview

The complete pipeline consists of 4 main stages:

1. **Finetuning Microsoft's BiomedParse**: Microsoft BiomedParse was finetuned on images of Fetal Heads that were publicly provided at the [HC18 Grand Challenge](https://zenodo.org/records/1327317). The model was trained to recognize "fetal head" as a text prompt for segmentation tasks.

2. **Segmentation**: The finetuned model performs slice-by-slice segmentation of the fetal head along the axial plane for all slices in the given ultrasound volume. Each 2D slice is processed individually using the text prompt "fetal head" to generate binary segmentation masks.

3. **Post-processing (Filtering)**: During the filtering phase, segmentation results are refined using geometric constraints:
   - Calculate ellipse measurements (major/minor axis lengths, centroid, orientation) for each segmented region
   - Use a reference slice (middle slice) to establish baseline measurements
   - Filter out regions that fall below threshold factors (typically 35-40% of reference measurements)
   - Remove duplicate regions per slice by keeping the one with maximum major axis length

4. **Reconstruction (Interpolation)**: During the interpolation phase, blank slices between the first and last filtered slices are filled:
   - Identify blank slices within the filtered range
   - Use the previous valid slice as a template
   - Apply adaptive scaling based on slice position relative to the volume center (±0.5% size adjustment)
   - Generate complete 3D volume with all gaps filled

## Output Formats
- **Raw segmentation**: Unfiltered slice-by-slice predictions
- **Filtered volume**: Geometrically consistent segmentation
- **Interpolated volume**: Complete 3D reconstruction with filled gaps
- All outputs saved as NIfTI (.nii.gz) format

## Steps to Reproduce

### 1. Environment Setup

#### Installation
```sh
git clone https://github.com/microsoft/BiomedParse.git
cd BiomedParse
```

[Notice] If inference_utils/target_dist.json is not cloned correctly, it will be automatically loaded from HuggingFace when needed.

#### Conda Environment Setup
**Option 1: Directly build the conda environment**
Under the project directory, run
```sh
conda env create -f environment.yml
```

**Option 2: Create a new conda environment from scratch**
```sh
conda create -n biomedparse python=3.9.19
conda activate biomedparse
```

Install Pytorch
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
In case there is issue with detectron2 installation, make sure your pytorch version is compatible with CUDA version on your machine at https://pytorch.org/.

Install dependencies
```sh
pip install -r assets/requirements/requirements.txt
```

### 2. Model Checkpoint

The finetuned model checkpoint is available [here](https://drive.google.com/file/d/1uqzYKx8g_bvrSB4ltWKyMF7-lrEdI5XW/view?usp=sharing).


### 3. (Optional) Finetuning on your own Dataset

Here are the instructions to Finetune BiomedParse on your own Data.

#### Raw Image and Annotation
Inputs: Images and Ground Truth Masks in 1024x1024 PNG format. For each dataset, put the raw image and mask files in the following format:
```
├── biomedparse_datasets
    ├── YOUR_DATASET_NAME
        ├── train
        ├── train_mask
        ├── test
        └── test_mask
```
Each folder should contain .png files. The mask files should be binary images where pixels != 0 indicates the foreground region.

#### File Name Convention
Each file name follows certain convention as:

[IMAGE-NAME]\_[MODALITY]\_[SITE].png

- [IMAGE-NAME] is any string that is unique for one image. The format can be anything.
- [MODALITY] is a string for the modality, such as "X-Ray"
- [SITE] is the anatomic site for the image, such as "chest"

One image can be associated with multiple masks corresponding to multiple targets in the image. The mask file name convention is:

[IMAGE-NAME]\_[MODALITY]\_[SITE]\_[TARGET].png

- [IMAGE-NAME], [MODALITY], and [SITE] are the same with the image file name.
- [TARGET] is the name of the target with spaces replaced by '+'. E.g. "tube" or "chest+tube". Make sure "_" doesn't appear in [TARGET].

#### Get Final Data File with Text Prompts
In biomedparse_datasets/create-customer-datasets.py, specify YOUR_DATASET_NAME. Run the script with:
```sh
cd biomedparse_datasets
python create-customer-datasets.py
```
After that, the dataset folder should be of the following format:
```
├── dataset_name
        ├── train
        ├── train_mask
        ├── train.json
        ├── test
        ├── test_mask
        └── test.json
```

#### Register Your Dataset for Training and Evaluation
In datasets/registration/register_biomed_datasets.py, simply add YOUR_DATASET_NAME to the datasets list. Registered datasets are ready to be added to the training and evaluation config file configs/biomed_seg_lang_v1.yaml. Your training dataset is registered as biomed_YOUR_DATASET_NAME_train, and your test dataset is biomed_YOUR_DATASET_NAME_test.

#### Train BiomedParse
To train the BiomedParse model, run:
```sh
bash assets/scripts/train.sh
```
This will continue train the model using the training datasets you specified in configs/biomed_seg_lang_v1.yaml

#### Evaluate BiomedParse
To evaluate the model, run:
```sh
bash assets/scripts/eval.sh
```
This will continue evaluate the model on the test datasets you specified in configs/biomed_seg_lang_v1.yaml.

## Inference

To run inference on a single ultrasound volume:

```bash
python inference.py
```

You will be prompted to provide the path to the 3D ultrasound volume. Upon providing the path, you will get the results in the default output directory.

**Disclaimer**: Before running inference.py, please ensure that you have updated the model checkpoint path in Line 59 of the inference.py file.

### Alternative: Step-by-Step Execution

For interactive processing,you can use the Jupyter notebook:

```bash
jupyter notebook EndToEndPipeline.ipynb
```

This notebook provides step-by-step execution of the complete pipeline.

## Acknowledgments

This work is built upon Microsoft's BiomedParse foundation model. For the original BiomedParse paper and implementation, please refer to:

**Original Repository**: [https://github.com/microsoft/BiomedParse](https://github.com/microsoft/BiomedParse)

We extend the original BiomedParse framework for fetal head segmentation and 3D reconstruction from ultrasound volumes.