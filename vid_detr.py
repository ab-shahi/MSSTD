import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pretrained DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# COCO class labels
CLASSES = [ 'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

# Colors
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Preprocessing
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Bounding box conversions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# Load video
video_path = r'/content/MVI_0790_VIS_OB.avi'  # Replace with your path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], pil_img.size)

    # Draw boxes
    for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), COLORS * 100):
        cls = p.argmax()
        label = f"{CLASSES[cls]}: {p[cls]:.2f}"
        color = tuple(int(x * 255) for x in c)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show output
    cv2.imshow('DETR on Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
frame_skip = 5  # process one frame every 5 frames
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # skip this frame


output_fps = 5  # 5 FPS output
out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    # run inference and display results


# sam2

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# Open video file
input_path = "input_video.mp4"
cap = cv2.VideoCapture(input_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video
output_fps = 5  # You can change this
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_sam_video.mp4", fourcc, output_fps, (width, height))

# Frame skip for FPS control
frame_skip = int(input_fps // output_fps) if output_fps < input_fps else 1
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    # Convert frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image_rgb)

    # Overlay masks on the frame
    for mask in masks:
        segmentation = mask['segmentation']
        color_mask = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        frame[segmentation] = cv2.addWeighted(frame[segmentation], 0.5, color_mask, 0.5, 0)

    # Write frame to output
    out.write(frame)
    frame_idx += 1

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()


import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# Load video
input_path = "input_video.mp4"
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_segmented_video.mp4", fourcc, fps, (width, height))

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

# Let user draw a bounding box on the first frame
bbox = cv2.selectROI("Select Object", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Convert BGR to RGB and set image
image_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# SAM expects bbox as [x0, y0, x1, y1]
input_box = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

# Predict mask
masks, scores, logits = predictor.predict(box=input_box[None, :], multimask_output=True)

# Choose the highest scoring mask
mask = masks[np.argmax(scores)]

# Apply mask to the first frame
colored_mask = np.zeros_like(first_frame)
colored_mask[mask] = [0, 255, 0]  # green
blended = cv2.addWeighted(first_frame, 0.7, colored_mask, 0.3, 0)

# Write the first frame with mask
out.write(blended)

# Process the rest of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply same box to all frames (optional: update box dynamically)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores, logits = predictor.predict(box=input_box[None, :], multimask_output=True)
    mask = masks[np.argmax(scores)]

    colored_mask = np.zeros_like(frame)
    colored_mask[mask] = [0, 255, 0]
    blended = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

    out.write(blended)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

