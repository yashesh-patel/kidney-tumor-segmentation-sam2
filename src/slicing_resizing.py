import nibabel as nib
import numpy as np
import cv2
import os

def load_nifti_data(nifti_path):
    img = nib.load(nifti_path)
    return img.get_fdata()

def segment_kidney_and_tumor(sagittal_slice):
    # Normalize the slice to [0, 255] and convert to uint8
    normalized_slice = cv2.normalize(sagittal_slice, None, 0, 255, cv2.NORM_MINMAX)
    normalized_slice = np.uint8(normalized_slice)
    
    # Apply Gaussian Blur to reduce noise
    blurred_slice = cv2.GaussianBlur(normalized_slice, (5, 5), 0)
    
    # Apply Canny edge detection (Ensure 8-bit image)
    edges = cv2.Canny(blurred_slice, 50, 150)
    
    # Apply morphological dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Watershed segmentation
    _, binary = cv2.threshold(blurred_slice, 100, 255, cv2.THRESH_BINARY)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(blurred_slice, cv2.COLOR_GRAY2BGR), markers)
    
    # Set kidney pixels to gray and tumor to white
    segmented_image = np.where(markers == 1, 128, 0).astype(np.uint8)  # Kidneys in gray
    segmented_image[markers == 2] = 255  # Tumors in white
    
    return segmented_image

def process_sagittal_slices(nifti_dir, output_dir, limit):
    case_folders = [case for case in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, case))]
    
    for i, case in enumerate(case_folders):
        if i >= limit:
            break
        nii_file = os.path.join(nifti_dir, case, 'imaging.nii.gz')
        if os.path.exists(nii_file):
            data = load_nifti_data(nii_file)
            num_slices_to_save = min(data.shape[0], 512)
            for slice_index in range(num_slices_to_save):
                sagittal_slice = data[slice_index, :, :]
                segmented_slice = segment_kidney_and_tumor(sagittal_slice)
                resized_slice = cv2.resize(segmented_slice, (256, 256), interpolation=cv2.INTER_AREA)
                slice_filename = f'{case}_sagittal_slice_{slice_index:03}.png'
                slice_path = os.path.join(output_dir, slice_filename)
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(slice_path, resized_slice)

nifti_dir = r'E:/kiTS23/kits23/dataset'
output_dir = r'E:/kiTS23/kits23/output4_slices'
process_sagittal_slices(nifti_dir, output_dir, 5)
