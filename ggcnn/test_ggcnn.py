import os
import cv2
import h5py
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from models.ggcnn import GGCNN
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp

def visualize_depth_map(depth_array, filename=None):
    # depth_array = np.array(depth_map)
    depth_min = depth_array.min()
    depth_max = depth_array.max()

    # print(f"Depth map values range: {depth_min} - {depth_max}")  # Print depth range for diagnosis

    # Apply logarithmic scaling to depth values
    scaled_depth_map = np.log1p(depth_array - depth_min) / np.log1p(depth_max - depth_min)

    # Replace invalid values with 0
    scaled_depth_map = np.nan_to_num(scaled_depth_map, nan=0, posinf=0, neginf=0)

    # import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(scaled_depth_map, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def process_depth_image(depth, crop_size=300, out_size=300, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape

    # Crop.
    depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                       (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    kernel = np.ones((3, 3),np.uint8)
    depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)

    depth_crop[depth_nan_mask==1] = 0

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale

    # Resize
    depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
        depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
        depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop

def crop_rgb(image):
    """
    Crop an RGB image from its center to a size of 300x300 pixels.

    Parameters:
        image_path (str): Path to the input image file.

    Returns:
        numpy.ndarray: Cropped image.
    """
    # Load image
    # image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to load the image.")
        return None

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate coordinates for cropping
    start_y = max(0, height // 2 - 150)  # Top left corner's y-coordinate
    end_y = min(height, height // 2 + 150)  # Bottom right corner's y-coordinate
    start_x = max(0, width // 2 - 150)  # Top left corner's x-coordinate
    end_x = min(width, width // 2 + 150)  # Bottom right corner's x-coordinate

    # Perform cropping
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

# depth image file path
dataset = "/home/dheeraj/ycb-dataset"
obj = "/004_sugar_box"
depth_file = "/NP3_0.h5"
rgb_file = "/NP3_0.jpg"

rgb_filepath = dataset + obj + rgb_file
depth_filepath =  dataset + obj + depth_file

# load test images
with h5py.File(depth_filepath, 'r') as f:
    depth = f['depth'][()].astype(float)

rgb = cv2.imread(rgb_filepath)
rgb_resize = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
rgb_crop = crop_rgb(rgb_resize)
cv2.imshow('RGB',rgb_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
# show image
# visualize_depth_map(depth)

# Load GGCNN with pretrained weights
model = GGCNN()

# path to weights
weights = os.path.dirname(os.path.abspath(__file__)) + "/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
model.load_state_dict(torch.load(weights))
model.eval()
# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Resize image to 300x300
resized_depth = process_depth_image(depth)
# visualize_depth_map(resized_depth)
cv2.imshow('Depth',resized_depth)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert numpy array to torch tensor
depth_img_tensor = torch.tensor(resized_depth).unsqueeze(0).unsqueeze(0).float()

# # Normalize the input image
# transform = transforms.Compose([
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# depth_img_tensor = transform(depth_img_tensor)

# Move the input tensor to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
depth_img_tensor = depth_img_tensor.to(device)

# Perform inference
with torch.no_grad():
    pos_output, cos_output, sin_output, width_output = model(depth_img_tensor)

# Post-process the outputs
q_img, ang_img, width_img = post_process_output(pos_output,cos_output,sin_output,width_output)

# plot outputs
evaluation.plot_output(rgb_crop,resized_depth, q_img, ang_img, no_grasps=1, grasp_width_img=width_img)