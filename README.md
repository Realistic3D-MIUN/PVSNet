<div align="center">
<a href="https://www.sciencedirect.com/science/article/pii/S0923596526002225"><img src='https://img.shields.io/badge/-Paper-FF6C00?style=flat&logo=elsevier&logoColor=white' alt='Paper'></a>
<a href='https://realistic3d-miun.github.io/PVSNet/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/3ZadeSSG/PVSNet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo_(View Synthesis)-blue'></a>
<a href='https://huggingface.co/spaces/3ZadeSSG/PLFNet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo_(Light Field Reconstruction)-blue'></a>
<a href='https://huggingface.co/datasets/3ZadeSSG/PVSNet_Blender_Dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
</div>

# PVSNet: Real-Time Position-Aware View Synthesis from Single-View Input.
This repo contains the supplementary materials for our paper PVSNet. The paper has been accepted, and we are starting to make the checkpoints available in the below table. The huggingface demo, dataset and project page links are in header of this README file.

# Setup
* Clone the repo and navigate into the directory
  ```bash
  git clone https://github.com/Realistic3D-MIUN/PVSNet.git
  cd PVSNet
  ```

* Download Checkpoints and place them into the `./checkpoint/` directory (We will opensource the model checkpoints for ablation study models as well, for instance Blender model with DINOv2 head):

| Model Checkpoint | Dataset | Resolution | Architecture | Download Link |
| --- | --- | --- | --- | --- |
| `checkpoint_blender_pvsnet_256x256.pth` | Blender | 256x256 | Regular | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_blender_pvsnet_256x256.pth) |
| `checkpoint_blender_pvsnet_512x512.pth` | Blender | 512x512 | Regular | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_blender_pvsnet_512x512.pth) |
| `checkpoint_blender_pvsnet_lite_256x256.pth` | Blender | 256x256 | Lite | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_blender_pvsnet_lite_256x256.pth) |
| `checkpoint_blender_pvsnet_lite_512x512.pth` | Blender | 512x512 | Lite | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_blender_pvsnet_lite_512x512.pth) |
| `checkpoint_coco_pvsnet_256x256.pth` | COCO | 256x256 | Regular | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_coco_pvsnet_256x256.pth) |
| `checkpoint_coco_pvsnet_lite_256x256.pth` | COCO | 256x256 | Lite | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_coco_pvsnet_lite_256x256.pth) |
| `checkpoint_coco_pvsnet_lite_512x512.pth` | COCO | 512x512 | Lite | [Link] |
| `checkpoint_coco_pvsnet_512x512.pth` | COCO | 512x512 | Regular | [Link] |
| `checkpoint_coco_pvsnet_256x256_T.pth` | COCO | 256x256 | Regular (Translation Only)| [Link] |
| `checkpoint_coco_pvsnet_512x512_T.pth` | COCO | 512x512 | Regular (Translation Only)| [Link] |
| `checkpoint_best_flowers.pth` | Flowers | Variable | Light Field | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_best_flowers.pth) |
| `checkpoint_best_stanford.pth` | Stanford | Variable | Light Field | [Link](https://huggingface.co/3ZadeSSG/PVSNet/blob/main/checkpoint_best_stanford.pth) |

### Dataset
The Blender data can be found at [Huggingface](https://huggingface.co/datasets/3ZadeSSG/PVSNet_Blender_Dataset)

# 1. PVSNet - Position-Aware View Synthesis
This model predicts novel views given an input image and target 3D/6D pose coordinates.

## 1.A. Gradio App (Interactive Demo)
Run the Gradio app to visualize the results in a web interface. You can generate videos with "Circle" or "Swing" trajectories.
```bash
python app_view_synthesis.py
```

## 1.B. Real-Time Inference
Run the mouse control script to interactively explore the view synthesis.
* **Arguments:**
    * `--dataset`: Choose between `blender` (default) and `coco`.
    * `--architecture`: Choose between `lite` (default) and `regular`.
    * `--resolution`: Choose between `256x256` (default) and `512x512`.
    * `--input_image`: Path to the input image.

* **Example Commands:**
  ```bash
  # Running with Regular Model (Blender Dataset):
  python render_with_gui.py --input_image ./sample_images/blender/blender_barbershop_12.png --architecture regular

  # Running with Lite Model (Blender Dataset):
  python render_with_gui.py --input_image ./sample_images/blender/blender_barbershop_12.png --architecture lite

  # Switching Architecture for COCO Model:
  python render_with_gui.py --dataset coco --input_image ./sample_images/real_world/bakery.jpeg --architecture lite
  ```

## 1.C. Faster Inference (TensorRT)
For maximum performance, you can use TensorRT. 

1. **Build the Engine:**
   First, build the TensorRT engines for all available models.
   ```bash
   python build_trt_pvsnet.py
   ```
   This will save `fp16` engines in the `./TRT_Engine/` directory.

2. **Run Inference with TRT UI:**
   Run the TensorRT-optimized interactive script. The arguments are the same as `render_with_gui.py`.
   ```bash
   # Running with Regular Model:
   python render_with_gui_trt.py --input_image ./sample_images/blender/blender_barbershop_12.png --architecture regular

   # Running with Lite Model:
   python render_with_gui_trt.py --input_image ./sample_images/blender/blender_barbershop_12.png --architecture lite

   # Switching Architecture for COCO Model:
   python render_with_gui_trt.py --dataset coco --input_image ./sample_images/real_world/bakery.jpeg --architecture lite
   ```

3. **Run Inference with TRT Gradio App:**
   ```bash
   python app_view_synthesis_trt.py
   ```

## 1.D. Single Image Prediction
Generate a single novel view from the command line:
```bash
python predict_view.py --image path/to/image.jpg --dataset coco --x 0.05 --y 0.02 --z -0.01 --checkpoint ./checkpoint/checkpoint_coco_pvsnet_lite_256x256.pth
```


# 2. PVSNet/PLFNet - Light Field Model
This model synthesizes dense light fields from a single input image. The demo file can be used to generate the videos, or nagivate the light fields in real-time post prediction. You can also run the model in real-time.

## 2.A. Gradio App (PyTorch)
Run the Gradio app for Light Field Reconstruction.
```bash
python app_light_field.py
```

## 2.B. Gradio App (TensorRT)
Run the TRT-optimized version for faster light field rendering (requires running `build_trt_pvsnet.py` first).
```bash
python app_light_field_trt.py
```

# 3. Testing and Evaluation

## 3.A. Testing Dataset Setup
To run evaluations natively, you need to download the Blender test dataset:
1. Download the test data zip file from [URL_HERE] and extract it.
2. Put the `lf_test.txt` file in the project root directory (`./PVSNet/lf_test.txt`).
3. Put the extracted dataset folder exactly one level up, inside the `dataset` directory (e.g. `../dataset/Blender_RGB`).

## 3.B. Run Evaluation
The testing script calculates MS-SSIM, SSIM, PSNR, LPIPS, DISTS, and VIF metrics on the fly.
```bash
python test.py --checkpoint ./checkpoint/checkpoint_blender_pvsnet_lite_512x512.pth
```
(Optional: add `--save_images` if you wish to export the predicted views to `./output/Blender/`)


## Supplementary Video
[![Watch the video](https://img.youtube.com/vi/ZZTcqOL4-WE/maxresdefault.jpg)](https://youtu.be/ZZTcqOL4-WE)

## Citation
If you use our work please use following citation:
```bibtex
@article{GOND2026117699,
title = {Real-time position-aware view synthesis from single-view input},
journal = {Signal Processing: Image Communication},
volume = {149},
pages = {117699},
year = {2026},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2026.117699},
url = {https://www.sciencedirect.com/science/article/pii/S0923596526002225},
author = {Manu Gond and Emin Zerman and Sebastian Knorr and Mårten Sjöström},
keywords = {View synthesis, Deep learning, Immersive imaging, Rendering, Position embedding, Light field}}
```

