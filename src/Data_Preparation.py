import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
import cv2
import shutil
from skimage.transform import resize
from PIL import Image
import albumentations as A
from tqdm import tqdm

# Load and normalize NIfTI data
def load_nifti_data(nifti_path):
    img = nib.load(nifti_path)
    try:
        data = img.get_fdata()
    except AttributeError:
        data = img.get_data()
    return data

def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data
    normalized_data = (data - mean) / std
    return normalized_data

# Process mask images for grayscale adjustments
def process_masks(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('L')
            colors = set(img.getdata())
            if len(colors) == 5:
                new_img = img.point(lambda p: 128 if p == 255 else p)
                new_img.save(os.path.join(output_folder, filename))
            else:
                img.save(os.path.join(output_folder, filename))

# Load and save slices with masks
def load_and_save_slices(nifti_dir, output_dir, mask_final_output_dir, limit=1):
    case_folders = [case for case in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, case))]
    for i, case in enumerate(case_folders):
        if i >= limit:
            break
        nii_file = os.path.join(nifti_dir, case, 'imaging.nii.gz')
        mask_file = os.path.join(nifti_dir, case, 'segmentation.nii.gz')
        if os.path.exists(nii_file) and os.path.exists(mask_file):
            print(f"Processing {nii_file} and {mask_file}...")
            data = load_nifti_data(nii_file)
            mask_data = load_nifti_data(mask_file)
            normalized_data = z_score_normalization(data)
            num_slices = normalized_data.shape[0]
            for slice_index in range(num_slices):
                mask_slice = mask_data[slice_index, :, :]
                if np.any(mask_slice > 0):
                    slice_data = normalized_data[slice_index, :, :]
                    resized_slice = resize(slice_data, (256, 256), anti_aliasing=True)
                    resized_mask = resize(mask_slice, (256, 256), anti_aliasing=False, preserve_range=True)
                    slice_filename = f'slice_{case}_{slice_index}.png'
                    mask_filename = f'slice_{case}_{slice_index}_mask.png'
                    slice_path = os.path.join(output_dir, slice_filename)
                    mask_path = os.path.join(mask_final_output_dir, mask_filename)
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(mask_final_output_dir, exist_ok=True)
                    plt.imsave(slice_path, resized_slice, cmap='gray')
                    plt.imsave(mask_path, resized_mask, cmap='gray')
                    print(f"Image and Mask Slice {slice_index} saved to {slice_path} and {mask_path}")
            process_masks(mask_final_output_dir, mask_final_output_dir)
        else:
            print(f"Files not found for {case}")

# Split data into train, validation, and test
def split_data(input_slices_dir, input_masks_dir, output_dir, train_size=0.7, val_size=0.15):
    output_dir = r'D:\Pawan\Project\ImgSeg\MedSam2\Medical-SAM2\Kidney & Kidney Tumor Segmentation\data\split_data'
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    for dir in [train_dir, val_dir, test_dir]:
        os.makedirs(dir, exist_ok=True)
        os.makedirs(os.path.join(dir, 'slices'), exist_ok=True)
        os.makedirs(os.path.join(dir, 'masks'), exist_ok=True)
    slice_files = [f for f in os.listdir(input_slices_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(input_masks_dir) if f.endswith('.png')]
    pairs = [(slice_file, mask_file) for slice_file, mask_file in zip(slice_files, mask_files)]
    random.seed(42)
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(train_size * total)
    val_end = train_end + int(val_size * total)
    train_pairs, val_pairs, test_pairs = pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]
    def copy_files(pairs, target_dir):
        for slice_file, mask_file in pairs:
            shutil.copy(os.path.join(input_slices_dir, slice_file), os.path.join(target_dir, 'slices', slice_file))
            shutil.copy(os.path.join(input_masks_dir, mask_file), os.path.join(target_dir, 'masks', mask_file))
    copy_files(train_pairs, train_dir)
    copy_files(val_pairs, val_dir)
    copy_files(test_pairs, test_dir)
    print("Data split completed!")

# Augmentation for slices and masks
def augment_images_and_masks(slices_dir, masks_dir, output_dir, num_augmentations=5):
    augmented_slices_dir = os.path.join(output_dir, 'augmented_slices')
    augmented_masks_dir = os.path.join(output_dir, 'augmented_masks')
    os.makedirs(augmented_slices_dir, exist_ok=True)
    os.makedirs(augmented_masks_dir, exist_ok=True)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomCrop(width=256, height=256, p=0.5),
    ])
    slice_files = [f for f in os.listdir(slices_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    assert len(slice_files) == len(mask_files), "Mismatch between slice and mask counts."
    for i in tqdm(range(len(slice_files))):
        slice_path = os.path.join(slices_dir, slice_files[i])
        mask_path = os.path.join(masks_dir, mask_files[i])
        slice_image = cv2.imread(slice_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        for j in range(num_augmentations):
            augmented = transform(image=slice_image, mask=mask_image)
            augmented_slice = augmented['image']
            augmented_mask = augmented['mask']
            slice_aug_name = f'slice_{slice_files[i].split("")[1]}{slice_files[i].split("")[2]}{j}.png'
            mask_aug_name = f'slice_{slice_files[i].split("")[1]}{slice_files[i].split("")[2]}_mask{j}.png'
            cv2.imwrite(os.path.join(augmented_slices_dir, slice_aug_name), augmented_slice)
            cv2.imwrite(os.path.join(augmented_masks_dir, mask_aug_name), augmented_mask)
    print("Augmentation completed!")

# Example usage:
nifti_dir = r'D:\Pawan\Project\ImgSeg\MedSam2\Medical-SAM2\Kidney & Kidney Tumor Segmentation\kits23\dataset'
output_dir = r'D:\Pawan\Project\ImgSeg\MedSam2\Medical-SAM2\Kidney & Kidney Tumor Segmentation\kits23\output_slices_Test'
mask_final_output_dir = r'D:\Pawan\Project\ImgSeg\MedSam2\Medical-SAM2\Kidney & Kidney Tumor Segmentation\kits23\output_masks_final'
split_output_dir = r'D:\Pawan\Project\ImgSeg\MedSam2\Medical-SAM2\Kidney & Kidney Tumor Segmentation\kits23\split_data'
load_and_save_slices(nifti_dir, output_dir, mask_final_output_dir, limit=5)
split_data(output_dir, mask_final_output_dir, split_output_dir)
augment_images_and_masks(os.path.join(split_output_dir, 'train/slices'), os.path.join(split_output_dir, 'train/masks'), split_output_dir, num_augmentations=5)