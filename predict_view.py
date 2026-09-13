import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import parameters as params
from models.pvsnet_model import PVSNet
import helperFunctions as helper

def getPositionVector(x, y, z, pitch=0, yaw=0, roll=0, pose_dims=6):
    if pose_dims == 3:
        vector = torch.zeros((1, 3), dtype=torch.float)
        vector[0][0] = (float(format(x, '.7f')) - (-0.1)) / (0.1 - (-0.1))
        vector[0][1] = (float(format(y, '.7f')) - (-0.1)) / (0.1 - (-0.1))
        vector[0][2] = (float(format(z, '.7f')) - (-0.1)) / (0.1 - (-0.1))
        return vector
    else:
        t_min, t_max = -0.1, 0.1
        r_min, r_max = -3, 3
        vector = torch.zeros((1, 6), dtype=torch.float)
        vector[0, 0] = (x - t_min) / (t_max - t_min)
        vector[0, 1] = (y - t_min) / (t_max - t_min)
        vector[0, 2] = (z - t_min) / (t_max - t_min)
        vector[0, 3] = (pitch - r_min) / (r_max - r_min)
        vector[0, 4] = (yaw - r_min) / (r_max - r_min)
        vector[0, 5] = (roll - r_min) / (r_max - r_min)
        return vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single novel view using PVSNet")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="output_view.png", help="Output image path")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/checkpoint_coco_pvsnet_lite_256x256.pth", help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, choices=["blender", "coco"], default="coco", help="Dataset type to determine pose dimensions (blender=3, coco=6)")
    parser.add_argument("--x", type=float, default=0.0, help="Translation X")
    parser.add_argument("--y", type=float, default=0.0, help="Translation Y")
    parser.add_argument("--z", type=float, default=0.0, help="Translation Z")
    parser.add_argument("--pitch", type=float, default=0.0, help="Rotation Pitch (only for COCO)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Rotation Yaw (only for COCO)")
    parser.add_argument("--roll", type=float, default=0.0, help="Rotation Roll (only for COCO)")
    
    args = parser.parse_args()

    # Parse config from checkpoint name or arguments
    base_name = os.path.basename(args.checkpoint).replace(".pth", "")
    is_lite = "lite" in base_name
    
    if "256x256" in base_name:
        height, width = 256, 256
    elif "512x512" in base_name:
        height, width = 512, 512
    else:
        height, width = 256, 256

    pose_dims = 3 if args.dataset == "blender" else 6

    print(f"Status: Loading Model (Resolution: {height}x{width}, Lite: {is_lite}, Pose Dims: {pose_dims})")
    
    model = PVSNet(total_image_input=params.params_number_input, pose_dims=pose_dims, height=height, width=width, is_lite=is_lite)
    model = helper.load_Checkpoint(args.checkpoint, model, load_cpu=True)
    model.to(params.DEVICE)
    model.eval()

    # Preprocess Input
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    
    img_input = Image.open(args.image).convert('RGB')
    img_input = transform(img_input).unsqueeze(0).to(params.DEVICE)
    
    pos = getPositionVector(args.x, args.y, args.z, args.pitch, args.yaw, args.roll, pose_dims=pose_dims).unsqueeze(0).to(params.DEVICE)

    # Inference
    print(f"Status: Generating view for Pose (x={args.x}, y={args.y}, z={args.z}, pitch={args.pitch}, yaw={args.yaw}, roll={args.roll})...")
    with torch.no_grad():
        predicted_img = model(img_input, pos)
        predicted_img = torch.clamp(predicted_img, 0, 1)

    save_image(predicted_img[0], args.output)
    print(f"Status: Saved predicted view to {args.output}")

