import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import SimpleBaseline model and configuration
import _init_paths
from models.pose_resnet import get_pose_net
from core.config import config, update_config
from core.inference import get_max_preds
from utils.vis import save_debug_images

# Load configuration and model
cfg_file = 'experiments/coco/resnet50/384x288_d256x3_adam_lr1e-3-RHPE-Foot-N2-Doctor-noflip-copy.yaml'
update_config(cfg_file)

model = get_pose_net(config, is_train=False)
model.eval()
model.init_weights(config.TEST.MODEL_FILE)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Image preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((384, 288)),  # 모델 입력 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    return image.unsqueeze(0)  # 배치 차원 추가


# Visualize keypoints on the image
def visualize_keypoints(image_path, keypoints, heatmap_shape):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image.shape)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x / heatmap_shape[0] * 512, y / heatmap_shape[1] * 407, s=100, c='red', marker='x')
        plt.text(x / heatmap_shape[0] * 512, y / heatmap_shape[1] * 407, str(i), fontsize=12, color='yellow')
    plt.show()


def get_keypoints_from_heatmap(heatmaps):
    heatmaps = heatmaps.squeeze().cpu().numpy()
    keypoints = []
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        keypoints.append((x, y))
    return keypoints, (heatmaps.shape[1:])


# Main processing function
def process_image(image_path):
    # Preprocess image
    input_image = preprocess_image(image_path)
    input_image = input_image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)

    # input_image = input_image.to("cpu")

    keypoints, heatmap_shape = get_keypoints_from_heatmap(output)
    visualize_keypoints(image_path, keypoints, heatmap_shape)


# Image path
image_path = 'data/coco/Foot/1.2.826.0.1.3680043.8.498.10561448436484761386439086564417613479-c.png'
process_image(image_path)
