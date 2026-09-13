# System
import os
import argparse
import time
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import torch
from torchvision.transforms import transforms
import torchvision
from models.pvsnet_model import PVSNet
import helperFunctions as helper
import parameters as params

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['blender', 'coco'], default='blender', help='Dataset type')
parser.add_argument('--resolution', type=str, choices=['256x256', '512x512'], default='256x256', help='Model resolution')
parser.add_argument('--architecture', type=str, choices=['regular', 'lite'], default='lite', help='Model architecture')
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mouse_sensitivity', type=int, default=5000, help="Set the mouse sensitivity")
parser.add_argument('--input_image', type=str, default="./sample_images/blender/blender_barbershop_12.png", help="Input image path")
opt, _ = parser.parse_known_args()

arch_str = "pvsnet_lite" if opt.architecture == "lite" else "pvsnet"
checkpoint_path = f"./checkpoint/checkpoint_{opt.dataset}_{arch_str}_{opt.resolution}.pth"
height, width = map(int, opt.resolution.split('x'))
pose_dims = 3 if opt.dataset == "blender" else 6
is_lite = (opt.architecture == "lite")

root = tk.Tk()
root.title(f"PVSNet Renderer ({opt.dataset}, {opt.resolution}, {opt.architecture})")
SCALE = opt.scale

print(f"Status: Loading Model from {checkpoint_path}")
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint {checkpoint_path} not found!")
    exit(1)

model = PVSNet(total_image_input=params.params_number_input, pose_dims=pose_dims, height=height, width=width, is_lite=is_lite)
model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
model.to(params.DEVICE)
model.eval()
print("Status: Model Loaded!")

transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor()
])

if not os.path.exists(opt.input_image):
    print(f"Error: Input image {opt.input_image} not found!")
    exit(1)

img_input = Image.open(opt.input_image).convert('RGB')
img_input = transform(img_input).unsqueeze(0).to(params.DEVICE)

def getPositionVector(x, y, z=0, pitch=0, yaw=0, roll=0):
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

current_translation = [0.0, 0.0, 0.0]  # x, y, z
current_rotation = [0.0, 0.0, 0.0]     # pitch, yaw, roll

def renderSingleFrame(x, y, z=0, pitch=0, yaw=0, roll=0):
    pos = getPositionVector(x, y, z, pitch, yaw, roll).unsqueeze(0).to(params.DEVICE)
    start_time = time.time()
    predicted_img = model(img_input, pos)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Render Time: {end_time - start_time:.4f}s | FPS: {1/(end_time-start_time+1e-6):.2f}")

    im = torchvision.transforms.functional.to_pil_image(torch.clamp(predicted_img[0], 0, 1))
    newsize = (width * SCALE, height * SCALE)
    im_resized = im.resize(newsize)
    return im_resized

def update_image():
    img = renderSingleFrame(
        current_translation[0],
        current_translation[1],
        current_translation[2],
        current_rotation[0],
        current_rotation[1],
        current_rotation[2]
    )
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

initial_img = renderSingleFrame(0,0)
img_tk = ImageTk.PhotoImage(initial_img)
label = tk.Label(root, image=img_tk)
label.image = img_tk
label.pack()

def on_left_drag(event):
    x_offset = (event.x - root.winfo_width() / 2) / opt.mouse_sensitivity
    y_offset = (event.y - root.winfo_height() / 2) / opt.mouse_sensitivity

    current_translation[0] = max(min(x_offset, 0.2), -0.2)
    current_translation[2] = max(min(y_offset, 0.2), -0.2)
    update_image()

def on_right_drag(event):
    if pose_dims == 6:
        rot_x = (event.y - root.winfo_height() / 2) / (opt.mouse_sensitivity / 50)
        rot_y = (event.x - root.winfo_width() / 2) / (opt.mouse_sensitivity / 50)

        current_rotation[0] = max(min(rot_x, 10), -10)
        current_rotation[2] = max(min(rot_y, 10), -10)
        update_image()

root.bind('<B1-Motion>', on_left_drag)
if pose_dims == 6:
    root.bind('<B3-Motion>', on_right_drag)

root.mainloop()
