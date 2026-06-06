import cv2
import os
# Annotations (class, x_center, y_center, width, height) – normalized
annotations = [''' labels for the image ''']
label_map = {1: 'a', 0: 'b', 4: 'c'}
colors = {1: (0, 255, 0), 0: (255, 0, 0), 4: (0, 0, 255)}  # BGR
# Load image (change path to your actual image file)
image_path = ''' image source '''
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
h, w = img.shape[:2]
for cls, cx, cy, bw, bh in annotations:
    x_center = cx * w
    y_center = cy * h
    box_width = bw * w
    box_height = bh * h
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 2)
    cv2.putText(img, label_map[cls], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[cls], 2)
# Save the result (no GUI display)
output_path = "Market photo/i.jpg"
cv2.imwrite(output_path, img)
print(f"Marked photo saved as: {output_path}")