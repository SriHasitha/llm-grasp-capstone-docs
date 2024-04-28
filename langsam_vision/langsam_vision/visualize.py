from PIL import Image
import h5py

import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_depth_map(depth_map, filename=None):
    depth_array = np.array(depth_map)
    depth_min = depth_array.min()
    depth_max = depth_array.max()

    print(f"Depth map values range: {depth_min} - {depth_max}")  # Print depth range for diagnosis

    # Apply logarithmic scaling to depth values
    scaled_depth_map = np.log1p(depth_array - depth_min) / np.log1p(depth_max - depth_min)

    # Replace invalid values with 0
    scaled_depth_map = np.nan_to_num(scaled_depth_map, nan=0, posinf=0, neginf=0)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(scaled_depth_map, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def save_depth_map(depth_map, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('depth', data=depth_map, compression='gzip', compression_opts=9)    

# def crop_images(rgb_image, depth_file, boxes, scale_x, scale_y):
#     rgb_image_np = np.array(rgb_image)
#     cropped_rgb_images = []
#     cropped_depth_images = []
#     crop_coordinates = []

#     with h5py.File(depth_file, 'r') as f:
#         depth_image_np = f['depth'][()].astype(np.float64)  # Convert depth values to float64
#         depth_height, depth_width = depth_image_np.shape

#         for box in boxes:
#             x_min, y_min, x_max, y_max = [int(x) for x in box]

#             # Convert bounding box coordinates from RGB image space to depth image space
#             depth_x_min = int(x_min * scale_x)
#             depth_y_min = int(y_min * scale_y)
#             depth_x_max = int(x_max * scale_x)
#             depth_y_max = int(y_max * scale_y)

#             # Ensure the coordinates are within the bounds of the depth image
#             depth_x_min = max(0, min(depth_x_min, depth_width))
#             depth_x_max = max(0, min(depth_x_max, depth_width))
#             depth_y_min = max(0, min(depth_y_min, depth_height))
#             depth_y_max = max(0, min(depth_y_max, depth_height))

#             # Crop RGB image
#             rgb_crop = rgb_image_np[y_min:y_max, x_min:x_max]
#             rgb_crop = Image.fromarray(rgb_crop)
#             cropped_rgb_images.append(rgb_crop)

#             # Crop depth image
#             depth_crop = depth_image_np[depth_y_min:depth_y_max, depth_x_min:depth_x_max]

#             if np.all(depth_crop == 0):
#                 raise ValueError("All depth values are zero in the cropped region")

#             # Convert depth crop to PIL Image without normalization
#             depth_crop = Image.fromarray(depth_crop.astype(np.uint16))
#             depth_crop = depth_crop.resize((500, 500))

#             cropped_depth_images.append(depth_crop)

#             # Store the crop coordinates
#             crop_coordinates.append((depth_x_min, depth_y_min, depth_x_max, depth_y_max))

#     return cropped_rgb_images, cropped_depth_images, crop_coordinates

def crop_images(rgb_image, depth_file, boxes, scale_x, scale_y):
    rgb_image_np = np.array(rgb_image)
    cropped_rgb_images = []
    cropped_depth_images = []
    crop_coordinates = []

    with h5py.File(depth_file, 'r') as f:
        depth_image_np = f['depth'][()].astype(np.float64)  # Convert depth values to float64
        depth_height, depth_width = depth_image_np.shape

        for box in boxes:
            x_min, y_min, x_max, y_max = [int(x) for x in box]

            # Calculate the center of the bounding box in the RGB image
            rgb_center_x = (x_min + x_max) // 2
            rgb_center_y = (y_min + y_max) // 2

            # Calculate the width and height of the bounding box in the RGB image
            rgb_box_width = x_max - x_min
            rgb_box_height = y_max - y_min
            print (rgb_box_width, rgb_box_height)
            # Convert bounding box coordinates from RGB image space to depth image space
            depth_center_x = int(rgb_center_x * scale_x)
            depth_center_y = int(rgb_center_y * scale_y)

            if (1.2*rgb_box_width) < (rgb_box_height):
                depth_crop_x_min = max(0, depth_center_x - 250)-100
                print("1")
            else:
                depth_crop_x_min = max(0, depth_center_x - 250)
                print("2")
            # Calculate the top-left coordinates of the 500x500 crop in the depth image
            depth_crop_y_min = max(0, depth_center_y - 250)
            depth_crop_x_max = min(depth_width, depth_center_x + 250)
            depth_crop_y_max = min(depth_height, depth_center_y + 250)

            # Adjust coordinates to ensure the crop fits entirely within the depth image
            depth_crop_width = depth_crop_x_max - depth_crop_x_min
            depth_crop_height = depth_crop_y_max - depth_crop_y_min

            if depth_crop_width < 500:
                if depth_center_x < 250:
                    depth_crop_x_min = 0
                    depth_crop_x_max = 500
                else:
                    depth_crop_x_max = depth_width
                    depth_crop_x_min = depth_width - 500
            if depth_crop_height < 500:
                if depth_center_y < 250:
                    depth_crop_y_min = 0
                    depth_crop_y_max = 500
                else:
                    depth_crop_y_max = depth_height
                    depth_crop_y_min = depth_height - 500

            # Crop RGB image
            rgb_crop = rgb_image_np[y_min:y_max, x_min:x_max]
            rgb_crop = Image.fromarray(rgb_crop)
            cropped_rgb_images.append(rgb_crop)

            # Crop depth image
            depth_crop = depth_image_np[depth_crop_y_min:depth_crop_y_max, depth_crop_x_min:depth_crop_x_max]

            if np.all(depth_crop == 0):
                raise ValueError("All depth values are zero in the cropped region")

            # Convert depth crop to PIL Image without normalization
            depth_crop = Image.fromarray(depth_crop.astype(np.uint16))
            depth_crop = depth_crop.resize((500, 500))

            cropped_depth_images.append(depth_crop)

            # Store the crop coordinates
            crop_coordinates.append((depth_crop_x_min, depth_crop_y_min, depth_crop_x_max, depth_crop_y_max))
            print(depth_crop_x_min, depth_crop_y_min, depth_crop_x_max, depth_crop_y_max)
            return cropped_rgb_images, cropped_depth_images, crop_coordinates
    

def visualize_output(image, masks, boxes, phrases, logits):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Visualize mask overlay on the image
    for mask in masks:
        image_with_mask = overlay_mask(image, mask)
    
    axes[0, 0].imshow(image_with_mask)
    axes[0, 0].set_title('Mask Overlay')
    axes[0, 0].axis('off')
    
    # Visualize box overlay on the image
    image_with_boxes = image.copy()
    for box in boxes:
        image_with_boxes = draw_box(image_with_boxes, box)
    
    axes[0, 1].imshow(image_with_boxes)
    axes[0, 1].set_title('Box Overlay')
    axes[0, 1].axis('off')

    # Display the phrase used
    phrases_str = '\n'.join(phrases)
    axes[1, 0].text(0, 0.5, phrases_str, fontsize=12, verticalalignment='center')
    axes[1, 0].set_title('Phrases')
    axes[1, 0].axis('off')
    
    # Display just the binary or grayscale mask
    for i, mask in enumerate(masks):
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('Binary Mask')
        axes[1, 1].axis('off')
        break  # Display only the first mask
        
    plt.tight_layout()
    plt.show()


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.3):
    image_np = np.array(image)  # Convert PIL image to NumPy array
    mask_np = mask.numpy()  # Convert TensorFlow tensor to NumPy array
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255  # Convert to binary mask
    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
    overlay = cv2.addWeighted(image_np, 1 - alpha, mask_np, alpha, 0)
    return overlay

def draw_box(image, box, color=(0, 255, 0), thickness=2):
    image_np = np.array(image)  # Convert PIL image to NumPy array
    start_point = (int(box[0]), int(box[1]))
    end_point = (int(box[2]), int(box[3]))
    return cv2.rectangle(image_np, start_point, end_point, color, thickness)

def visualize_output_depth(cropped_rgb_images, cropped_depth_images, rgb_image, boxes):
    num_images = len(cropped_rgb_images)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    for i in range(num_images):
        rgb_image_crop = cropped_rgb_images[i]
        depth_image = cropped_depth_images[i]
        original_image_with_box = np.array(rgb_image.copy())  # Convert PIL Image to numpy array

        if num_images == 1:
            rgb_ax = axes[0]
            depth_ax = axes[1]
            cropped_ax = axes[2]
        else:
            rgb_ax = axes[i, 0]
            depth_ax = axes[i, 1]
            cropped_ax = axes[i, 2]

        # Overlay bounding box on original image
        box = boxes[i]
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        cv2.rectangle(original_image_with_box, start_point, end_point, (0, 255, 0), 2)

        # Calculate crop coordinates for the RGB image
        crop_width = 1000
        crop_height = 1066
        center_x = (start_point[0] + end_point[0]) / 2
        center_y = (start_point[1] + end_point[1]) / 2
        crop_x_min = max(0, int(center_x - crop_width / 2))
        crop_x_max = min(rgb_image.width, int(center_x + crop_width / 2))
        crop_y_min = max(0, int(center_y - crop_height / 2))
        crop_y_max = min(rgb_image.height, int(center_y + crop_height / 2))

        # Crop the region from the original image
        cropped_image = rgb_image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

        # Calculate crop coordinates for the depth image
        depth_aspect_ratio = depth_image.width / depth_image.height
        depth_crop_height = crop_height
        depth_crop_width = int(depth_crop_height * depth_aspect_ratio)
        
        # Apply shift for correct cropping in the depth image
        depth_shift = 0  # Adjust as neededa
        depth_center_x = center_x * (depth_image.width / rgb_image.width) + depth_shift
        depth_center_y = center_y * (depth_image.height / rgb_image.height)
        depth_crop_x_min = max(0, int(depth_center_x - depth_crop_width / 2))
        depth_crop_x_max = min(depth_image.width, int(depth_center_x + depth_crop_width / 2))
        depth_crop_y_min = max(0, int(depth_center_y - depth_crop_height / 2))
        depth_crop_y_max = min(depth_image.height, int(depth_center_y + depth_crop_height / 2))

        # Crop the region from the depth image
        cropped_depth = depth_image.crop((depth_crop_x_min, depth_crop_y_min, depth_crop_x_max, depth_crop_y_max))

        # Display original image with bounding box
        rgb_ax.imshow(original_image_with_box)
        rgb_ax.set_title('Original Image with Box')
        rgb_ax.axis('off')

        # Display cropped depth image
        depth_ax.imshow(cropped_depth, cmap='viridis', vmin=0, vmax=1)
        depth_ax.set_title('Cropped Depth Image')
        depth_ax.axis('off')

        # Display cropped image from original image
        cropped_ax.imshow(cropped_image)
        cropped_ax.set_title('Cropped Image')
        cropped_ax.axis('off')

    plt.tight_layout()
    plt.show()
