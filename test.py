import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import parameters as params
from models.pvsnet_model import PVSNet
import helperFunctions as helper

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, VisualInformationFidelity
import lpips as LearnedPerceptualImagePatchSimilarity
from DISTS_pytorch import DISTS as DeepImageStructureandTextureSimilarity

def getPositionVector(x, y, z):
    vector = torch.zeros((1, 3), dtype=torch.float)
    normalized_x = (float(format(x, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_y = (float(format(y, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_z = (float(format(z, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    vector[0][0] = normalized_x
    vector[0][1] = normalized_y
    vector[0][2] = normalized_z
    return vector

def evaluate(model, test_file_location, output_location, height, width, save_images=False):
    lists_and_labels = np.loadtxt(test_file_location, dtype=str)
    _input_rgb = lists_and_labels[:, 0]
    _output_rgb = lists_and_labels[:, 4]
    _output_x = lists_and_labels[:, 5]
    _output_y = lists_and_labels[:, 6]
    _output_z = lists_and_labels[:, 7]

    transform_input = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    transform_gt = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])

    MS_SSIM = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(params.DEVICE)
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(params.DEVICE)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(params.DEVICE)
    LPIPS = LearnedPerceptualImagePatchSimilarity.LPIPS().to(params.DEVICE)
    DISTS = DeepImageStructureandTextureSimilarity().to(params.DEVICE)
    VIF = VisualInformationFidelity().to(params.DEVICE)

    ms_ssim_list, ssim_list, psnr_list = [], [], []
    lpips_list, dists_list, vif_list = [], [], []

    print(f"Starting evaluation of {len(_input_rgb)} samples...")

    for index in range(len(_input_rgb)):
        img_input = Image.open(_input_rgb[index]).convert('RGB')
        img_input = transform_input(img_input).unsqueeze(0).to(params.DEVICE)

        x = float(_output_x[index])
        y = float(_output_y[index])
        z = float(_output_z[index])
        output_position = getPositionVector(x, y, z).unsqueeze(0).to(params.DEVICE)

        # Generate prediction
        with torch.no_grad():
            img_out = model(img_input, output_position)
            img_out = torch.clamp(img_out, 0, 1)

        output_loc = _output_rgb[index]
        filename = output_loc.split("/")[-1]
        dirname = output_loc.split("/")[-2]

        if save_images:
            os.makedirs(os.path.join(output_location, dirname), exist_ok=True)
            save_image(img_out, os.path.join(output_location, dirname, filename))

        # Load GT
        gt_img = Image.open(output_loc).convert('RGB')
        gt_tensor = transform_gt(gt_img).unsqueeze(0).to(params.DEVICE)

        # Metrics
        ms_ssim_val = MS_SSIM(img_out, gt_tensor).item()
        ssim_val = SSIM(img_out, gt_tensor).item()
        psnr_val = PSNR(img_out, gt_tensor).item()
        lpips_val = LPIPS(img_out, gt_tensor).item()
        dists_val = DISTS(img_out, gt_tensor).item()
        vif_val = VIF(img_out, gt_tensor).item()

        ms_ssim_list.append(ms_ssim_val)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        lpips_list.append(lpips_val)
        dists_list.append(dists_val)
        vif_list.append(vif_val)

        print(f"Processing: {index + 1}/{len(_input_rgb)} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}", end="\r")

    print("\n\n--- Evaluation Results ---")
    print(f"MS_SSIM: {np.mean(ms_ssim_list):.4f}")
    print(f"SSIM:    {np.mean(ssim_list):.4f}")
    print(f"PSNR:    {np.mean(psnr_list):.4f}")
    print(f"LPIPS:   {np.mean(lpips_list):.4f}")
    print(f"DISTS:   {np.mean(dists_list):.4f}")
    print(f"VIF:     {np.mean(vif_list):.4f}")
    print("--------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PVSNet on Blender Dataset")
    parser.add_argument("--test_file", type=str, default="./lf_test.txt", help="Path to lf_test.txt")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/checkpoint_blender_pvsnet_lite_256x256.pth", help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./output/Blender/", help="Where to save predicted images if enabled")
    parser.add_argument("--save_images", action="store_true", help="Save predicted images to disk")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse config from checkpoint name
    base_name = os.path.basename(args.checkpoint).replace(".pth", "")
    parts = base_name.split('_')
    is_lite = "lite" in base_name
    res_str = parts[-1]
    h_str, w_str = res_str.split('x')
    height, width = int(h_str), int(w_str)
    
    # Force pose_dims to 3 since this script is specifically for Blender evaluation
    pose_dims = 3

    print(f"Loading Model: Resolution {height}x{width}, Lite={is_lite}")
    model = PVSNet(total_image_input=params.params_number_input, pose_dims=pose_dims, height=height, width=width, is_lite=is_lite)
    model = helper.load_Checkpoint(args.checkpoint, model, load_cpu=True)
    model.to(params.DEVICE)
    model.eval()

    evaluate(model, args.test_file, args.output_dir, height, width, save_images=args.save_images)
