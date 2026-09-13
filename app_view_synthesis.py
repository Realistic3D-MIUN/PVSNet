import gradio as gr
import torch
import numpy as np
import cv2
import tempfile
from PIL import Image
import torchvision.transforms as transforms
import os
from huggingface_hub import hf_hub_download
import huggingface_hub

from models.pvsnet_model import PVSNet
import helperFunctions as helper
import parameters as params

DEVICE = params.DEVICE

def getPositionVector(x, y, z, pose_dims=3):
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
        vector[0, 3] = (0 - r_min) / (r_max - r_min)
        vector[0, 4] = (0 - r_min) / (r_max - r_min)
        vector[0, 5] = (0 - r_min) / (r_max - r_min)
        return vector

def generateCircularTrajectory(radius, num_frames):
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    return [[radius * np.cos(angle), radius * np.sin(angle), 0] for angle in angles]

def generateSwingTrajectory(radius, num_frames):
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    return [[radius * np.cos(angle), 0, radius * np.sin(angle)] for angle in angles]

def create_video_from_memory(frames, fps=30):
    if not frames:
        return None
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return temp_video.name

def process_image(img, video_type, radius, num_frames, num_loops, dataset, resolution, architecture):
    if img is None:
        return None
        
    width, height = map(int, resolution.split('x'))

    min_dim = min(img.width, img.height)
    left = (img.width - min_dim) / 2
    top = (img.height - min_dim) / 2
    right = (img.width + min_dim) / 2
    bottom = (img.height + min_dim) / 2
    img = img.crop((left, top, right, bottom))

    is_lite = (architecture == "Lite")
    pose_dims = 3 if dataset == "Blender" else 6
    
    dataset_prefix = dataset.lower()
    arch_infix = "pvsnet_lite" if is_lite else "pvsnet"
    checkpoint_name = f"checkpoint_{dataset_prefix}_{arch_infix}_{resolution}.pth"
    checkpoint_path = os.path.join("./checkpoint", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise gr.Error(f"Checkpoint {checkpoint_name} not found! Please select a valid combination.")

    model = PVSNet(total_image_input=params.params_number_input, pose_dims=pose_dims, height=height, width=width, is_lite=is_lite)
    
    try:
        model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise gr.Error(f"Error loading checkpoint {checkpoint_path}: {e}")
        
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    
    img_input = img.convert('RGB')
    img_input = transform(img_input).unsqueeze(0).to(DEVICE)

    if video_type == "Circle":
        raw_traj = generateCircularTrajectory(radius, num_frames)
        trajectory = [(p[0], p[1], 0) for p in raw_traj]
    elif video_type == "Swing":
        raw_traj = generateSwingTrajectory(radius, num_frames)
        trajectory = raw_traj
    else:
         raw_traj = generateCircularTrajectory(radius, num_frames)
         trajectory = [(p[0], p[1], 0) for p in raw_traj]

    view_frames = []

    for x, y, z in trajectory:
        pos = getPositionVector(x, y, z, pose_dims=pose_dims).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predicted_img = model(img_input, pos)
        
        p_img = predicted_img[0].detach().cpu().permute(1, 2, 0).numpy()
        p_img = np.clip(p_img, 0, 1)
        p_img = (p_img * 255).astype(np.uint8)
        p_img_bgr = cv2.cvtColor(p_img, cv2.COLOR_RGB2BGR)
        view_frames.append(p_img_bgr)

    view_frames = view_frames * int(num_loops)

    fps = 60
    view_video_path = create_video_from_memory(view_frames, fps=fps)

    return view_video_path

with gr.Blocks(title="PVSNet", theme="default") as demo:
    gr.Markdown(
    """
    ## PVSNet: Real-Time Position-Aware View Synthesis from Single-View Input
    * Upload an image and get a mini video showing capability of novel view synthesis.
    **Note:** Huggingface demo is running on CPU so inference speeds will be slow. Inference might take around 2-5 mins depending on resolution and model. We recomment runnning it on 256x256 and with Lite model.
    ### Head to our [Project Page](https://realistic3d-miun.github.io/PVSNet/) for more details about the models
    """)
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Input Image", height=256)
            
            with gr.Group():
                dataset_type = gr.Dropdown(["Blender", "COCO"], label="Dataset Model", value="COCO")
                resolution_type = gr.Dropdown(["256x256", "512x512"], label="Resolution", value="256x256")
                architecture_type = gr.Dropdown(["Regular", "Lite"], label="Architecture", value="Lite")
                video_type = gr.Dropdown(["Circle", "Swing"], label="Trajectory Type", value="Swing")
            
            with gr.Accordion("Advanced Settings", open=False):
                radius = gr.Slider(0.01, 0.1, value=0.06, label="Motion Radius")
                num_frames = gr.Slider(10, 120, value=60, step=1, label="Frames per Loop")
                num_loops = gr.Slider(1, 6, value=3, step=1, label="Number of Loops")

            
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated View Video", height=256)

    submit_btn.click(
        fn=process_image,
        inputs=[img_input, video_type, radius, num_frames, num_loops, dataset_type, resolution_type, architecture_type],
        outputs=[video_output]
    )

    gr.Markdown("### Example Images: Click to Load")
    
    import glob
    blender_imgs = sorted(glob.glob("./sample_images/blender/*"))
    coco_imgs = sorted(glob.glob("./sample_images/coco/*"))
    rw_imgs = sorted(glob.glob("./sample_images/real_world/*"))

    def create_grid(imgs, title, cols=4):
        gr.Markdown(title)
        image_components = []
        for i in range(0, len(imgs), cols):
            with gr.Row():
                for img_path in imgs[i:i+cols]:
                    comp = gr.Image(img_path, label=os.path.basename(img_path), height=150, interactive=False, show_label=True)
                    image_components.append((comp, img_path))
        return image_components

    with gr.Column():
        b_comps = create_grid(blender_imgs, "#### Blender Models (Loads Blender Lite 256x256 by Default)")
        c_comps = create_grid(coco_imgs, "#### COCO Models (Loads COCO 256x256 Lite by Default)")
        r_comps = create_grid(rw_imgs, "#### Real World Models (Loads COCO 256x256 Lite by Default)")

    for comp, path in b_comps:
        comp.select(fn=lambda p=path: (Image.open(p), "Blender", "256x256", "Lite"), outputs=[img_input, dataset_type, resolution_type, architecture_type])
    
    for comp, path in c_comps:
        comp.select(fn=lambda p=path: (Image.open(p), "COCO", "256x256", "Lite"), outputs=[img_input, dataset_type, resolution_type, architecture_type])

    for comp, path in r_comps:
        comp.select(fn=lambda p=path: (Image.open(p), "COCO", "256x256", "Lite"), outputs=[img_input, dataset_type, resolution_type, architecture_type])

if __name__ == "__main__":
    demo.launch()
