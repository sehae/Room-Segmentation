from SAM2Source.sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from SAM2Source.sam2.sam2.build_sam import build_sam2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

# Environment setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SAM2 model
sam2_checkpoint = "./SAM2Source/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# Load image
image_path = 'floor_plan.jpg'  # Change this to your image file
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess the image (thresholding, edge detection)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Detect corners using Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(edges, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int32(corners)  # Fix the conversion to integer type

# Draw corners on the image
corner_image = image.copy()
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(corner_image, (x, y), 5, (0, 255, 0), -1)

# Display the corners
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Corners")
plt.axis("off")
plt.show()

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area (to ignore small noise)
min_area = 1000  # Adjust based on your image size
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw bounding boxes around detected contours
boxes = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    boxes.append({"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h})
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display contours on the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Contours")
plt.axis("off")
plt.show()

# Generate masks for detected regions
def generate_masks_for_boxes(image, boxes, mask_generator):
    masks = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box.values()
        cropped_image = image[y_min:y_max, x_min:x_max]
        box_masks = mask_generator.generate(cropped_image)
        # Adjust mask coordinates back to the original image
        for mask in box_masks:
            mask['segmentation'] = np.pad(mask['segmentation'],
                                          [(y_min, image.shape[0] - y_max),
                                           (x_min, image.shape[1] - x_max)])
            masks.append(mask)
    return masks

room_masks = generate_masks_for_boxes(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes, mask_generator)

# Display segmented results
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)

plt.figure(figsize=(20, 20))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
show_anns(room_masks)
plt.axis('off')
plt.title("Segmented Areas")
plt.show()
