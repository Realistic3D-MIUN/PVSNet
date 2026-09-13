import os
import subprocess
import torch
import torch.onnx
import glob
from models.pvsnet_model import PVSNet
from models.pvsnet_light_field_model import PLFNet
import parameters as params
import helperFunctions as helper

# Configuration
ONNX_DIR = "./checkpoint_onnx"
ENGINE_DIR = "./TRT_Engine"

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(ENGINE_DIR, exist_ok=True)

def export_pvsnet_to_onnx(checkpoint_path, output_name, pose_dims, height, width, is_lite):
    print(f"\nStatus: Loading PVSNet {output_name} Model from {checkpoint_path}")
    model = PVSNet(total_image_input=params.params_number_input, pose_dims=pose_dims, height=height, width=width, is_lite=is_lite)
    model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
    model.eval()
    model.cuda()
    print(f"Status: {output_name} Model Loaded")

    dummy_img = torch.randn(1, 3, height, width).cuda()
    dummy_pos = torch.randn(1, pose_dims).cuda()

    onnx_path = os.path.join(ONNX_DIR, f"{output_name}.onnx")
    print(f"Status: Exporting ONNX to {onnx_path}...")
    
    torch.onnx.export(
        model,
        (dummy_img, dummy_pos),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_image', 'input_pos'],
        output_names=['output_image']
    )
    print(f"Status: ONNX exported to {onnx_path}")
    return onnx_path


def export_plfnet_to_onnx(checkpoint_path, output_name, height, width):
    print(f"\nStatus: Loading PLFNet {output_name} Model from {checkpoint_path}")
    model = PLFNet()
    model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
    model.eval()
    model.cuda()
    print(f"Status: {output_name} Model Loaded")

    dummy_img_cat = torch.randn(1, 5, height, width).cuda()

    onnx_path = os.path.join(ONNX_DIR, f"{output_name}.onnx")
    print(f"Status: Exporting ONNX to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_img_cat,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_image_cat'],
        output_names=['output_image']
    )
    print(f"Status: ONNX exported to {onnx_path}")
    return onnx_path


def build_engine(onnx_path, engine_name):
    engine_path = os.path.join(ENGINE_DIR, f"{engine_name}.engine")
    if os.path.exists(engine_path):
        print(f"Status: Engine {engine_path} already exists. Skipping build.")
        return

    print(f"Status: Building TensorRT engine for {engine_name}...")
    try:
        subprocess.run([
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16"
        ], check=True)
        print(f"Status: TensorRT engine saved to {engine_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error building engine: {e}")
    except FileNotFoundError:
        print("Error: 'trtexec' not found in PATH. Please make sure TensorRT is installed.")

def main():
    checkpoints = glob.glob("./checkpoint/*.pth")
    for cp in checkpoints:
        base_name = os.path.basename(cp).replace(".pth", "")
        
        if "flowers" in base_name or "stanford" in base_name:
            # We will export for 176x256 as default for TensorRT
            h, w = 176, 256
            onnx_path = export_plfnet_to_onnx(cp, base_name, h, w)
            build_engine(onnx_path, base_name + "_fp16")
        else:
            # Parse PVSNet config
            # e.g. checkpoint_blender_pvsnet_lite_256x256
            parts = base_name.split('_')
            dataset = parts[1] # blender or coco
            
            is_lite = "lite" in base_name
            res_str = parts[-1]
            h_str, w_str = res_str.split('x')
            h, w = int(h_str), int(w_str)
            pose_dims = 3 if dataset == "blender" else 6
            
            onnx_path = export_pvsnet_to_onnx(cp, base_name, pose_dims, h, w, is_lite)
            build_engine(onnx_path, base_name + "_fp16")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()

