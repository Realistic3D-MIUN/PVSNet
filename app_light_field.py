import gradio as gr
import torch
import numpy as np
import cv2
import io
import tempfile
import base64
from PIL import Image
import torchvision.transforms as transforms
import parameters as params
from models.pvsnet_light_field_model import PLFNet
import helperFunctions as helper
import socket
import os
import json
import joblib

MODEL_FLOWERS_LOCATION = "./checkpoint/checkpoint_best_flowers.pth"
MODEL_STANFORD_LOCATION = "./checkpoint/checkpoint_best_stanford.pth"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATASET_CHECKPOINT_MAP = {
    "Flowers": MODEL_FLOWERS_LOCATION,
    "Stanford": MODEL_STANFORD_LOCATION,
}

SAMPLE_IMAGE_DIR = "./sample_images/Light_Field"
SAMPLE_IMAGES = {}
for dataset_name in ["Flowers", "Stanford"]:
    folder = os.path.join(SAMPLE_IMAGE_DIR, dataset_name)
    if os.path.isdir(folder):
        images = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        SAMPLE_IMAGES[dataset_name] = images

def getPositionVector(x, y, height, width):
    vector = torch.zeros((2, height, width), dtype=torch.float)
    normalized_x = (x - (-0.003)) / (0.003 - (-0.003))
    normalized_y = (y - (-0.003)) / (0.003 - (-0.003))
    vector[0, :, :] = normalized_x
    vector[1, :, :] = normalized_y
    return vector

def predictSingleImage(model, img, target_pose, height, width):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    img_input = transform(img).to(DEVICE)
    output_position = getPositionVector(target_pose[0], target_pose[1], height, width).to(DEVICE)
    with torch.no_grad():
        img_ = torch.cat((img_input, output_position), dim=0).unsqueeze(0).to(DEVICE)
        img_out = model(img_).detach().cpu()
    return img_out

def generateCircularTrajectory(radius, num_frames, num_loops):
    angles = np.linspace(0, 2 * np.pi * num_loops, num_frames * num_loops)
    return [[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles]

def create_video_from_memory(frames, fps=60):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return temp_video.name

def process_parallax_video(img, dataset, resolution, radius, num_frames, num_loops):
    if img is None:
        return None
    checkpoint_path = DATASET_CHECKPOINT_MAP.get(dataset, DATASET_CHECKPOINT_MAP["Flowers"])
    model = PLFNet()
    model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
    model.to(DEVICE)
    model.eval()

    height, width = (352, 512) if "352x512" in resolution else (176, 256)
    img = img.crop((0, 0, img.width, int(img.width * (11 / 16))))
    trajectory = generateCircularTrajectory(radius, num_frames, num_loops)

    frames = []
    for pose in trajectory:
        output_img = predictSingleImage(model, img, pose, height, width)
        img_np = output_img.squeeze(0).permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        frames.append(img_bgr)

    return create_video_from_memory(frames)

def generate_lf_raw_frames(img, dataset, resolution):
    if img is None:
        return None, "Please upload an image first."
        
    checkpoint_path = DATASET_CHECKPOINT_MAP.get(dataset, DATASET_CHECKPOINT_MAP["Flowers"])
    model = PLFNet()
    model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
    model.to(DEVICE)
    model.eval()

    height, width = (352, 512) if "352x512" in resolution else (176, 256)
    img = img.crop((0, 0, img.width, int(img.width * (11 / 16))))
    
    frames_b64 = []
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            pose = [i * 0.001, j * 0.001]
            out = predictSingleImage(model, img, pose, height, width)
            img_np = out.squeeze(0).permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(f"data:image/jpeg;base64,{b64_str}")
            
    import json
    return json.dumps(frames_b64), "Light Field generated! You can now adjust Focus and Aperture, and move your mouse over the image."



html_code = """
<iframe id="lf-iframe" style="width: 100%; max-width: 800px; aspect-ratio: 512/352; border: 2px dashed #ccc; display: block; margin: 0 auto; background: #222;" srcdoc="
<html>
<head>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>
  <script src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'></script>
  <style> body { margin: 0; overflow: hidden; background: #222; } canvas { display: block; width: 100%; height: 100%; } #placeholder { color: #888; font-family: sans-serif; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); pointer-events: none; } #debug { position: absolute; top: 0; left: 0; color: lime; font-family: monospace; padding: 10px; pointer-events: none; z-index: 9999; } </style>
</head>
<body>
  <div id='placeholder'>Generated Light Field will appear here</div>
  <div id='debug'></div>
  <script>
    function logDebug(msg) {
        document.getElementById('debug').innerHTML += msg + '<br>';
    }

    const vertexShader = `
      out vec2 vSt;
      out vec2 vUv;
      void main() {
          vec3 posToCam = cameraPosition - position;
          vec3 nDir = normalize(posToCam);
          float zRatio = posToCam.z / nDir.z;
          vec3 uvPoint = zRatio * nDir;
          vUv = uvPoint.xy + 0.5;
          vUv.x = 1.0 - vUv.x;
          vSt = uv;
          vSt.x = 1.0 - vSt.x;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      precision highp sampler2DArray;
      uniform sampler2DArray field;
      uniform vec2 camArraySize;
      uniform float aperture;
      uniform float focus;
      in vec2 vSt;
      in vec2 vUv;
      out vec4 fragColor;
      
      void main() {
          vec4 color = vec4(0.0);
          float colorCount = 0.0;
          if (vUv.x < 0.0 || vUv.x > 1.0 || vUv.y < 0.0 || vUv.y > 1.0) {
              discard;
          }
          for (float i = 0.0; i < 7.0; i++) {
              for (float j = 0.0; j < 7.0; j++) {
                  float dx = i - (vSt.x * camArraySize.x - 0.5);
                  float dy = j - (vSt.y * camArraySize.y - 0.5);
                  float sqDist = dx * dx + dy * dy;
                  if (sqDist <= aperture + 0.001) {
                      float camOff = i + camArraySize.x * j;
                      vec2 focOff = vec2(dx, dy) * focus;
                      color += texture(field, vec3(vUv + focOff, camOff));
                      colorCount++;
                  }
              }
          }
          fragColor = vec4(color.rgb / max(colorCount, 1.0), 1.0);
      }
    `;

    let scene, camera, renderer, planeMat, fieldTexture, controls;
    let camsX = 7, camsY = 7;
    let reqFrame;

    function initScene() {
        if(scene) return;
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(0, 0, 2);
        camera.lookAt(0, 0, -2);

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('webgl2', { antialias: true });
        if (!context) logDebug('WebGL2 not supported!');
        renderer = new THREE.WebGLRenderer({ canvas: canvas, context: context });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.set(0, 0, -2);
        controls.panSpeed = 2;

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    function render() {
        if (controls) controls.update();
        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
        reqFrame = requestAnimationFrame(render);
    }

    window.addEventListener('message', async function(e) {
        if (e.data.type === 'update_frames') {
            document.getElementById('placeholder').innerText = 'Loading textures...';
            document.getElementById('placeholder').style.display = 'block';
            
            const frames = e.data.frames;
            if (!frames || frames.length !== 49) {
                logDebug('Error: Invalid frames');
                return;
            }
            
            try {
                initScene();
                if (controls) {
                    controls.rotateSpeed = e.data.sensitivity;
                    controls.panSpeed = e.data.sensitivity * 2;
                }
                
                let resX, resY;
                const allBuffer = [];
                
                for(let i=0; i<49; i++) {
                    const img = new Image();
                    await new Promise((resolve, reject) => { 
                        img.onload = resolve; 
                        img.onerror = reject;
                        img.src = frames[i]; 
                    });
                    if(i===0) { resX = img.width; resY = img.height; }
                    const cvs = document.createElement('canvas');
                    cvs.width = resX; cvs.height = resY;
                    const ctx = cvs.getContext('2d', { willReadFrequently: true });
                    ctx.drawImage(img, 0, 0);
                    const data = ctx.getImageData(0, 0, resX, resY).data;
                    allBuffer.push(data);
                }
                
                const totalBuffer = new Uint8Array(resX * resY * 4 * 49);
                for(let i=0; i<49; i++) {
                    totalBuffer.set(allBuffer[i], i * resX * resY * 4);
                }

                if(planeMat) {
                    scene.remove(scene.children[0]);
                    planeMat.dispose();
                }

                fieldTexture = new THREE.DataTexture2DArray(totalBuffer, resX, resY, 49);
                fieldTexture.format = THREE.RGBAFormat;
                fieldTexture.type = THREE.UnsignedByteType;
                fieldTexture.needsUpdate = true;
                
                planeMat = new THREE.ShaderMaterial({
                    uniforms: {
                        field: { value: fieldTexture },
                        camArraySize: { value: new THREE.Vector2(camsX, camsY) },
                        aperture: { value: e.data.aperture },
                        focus: { value: e.data.focus }
                    },
                    vertexShader: vertexShader,
                    fragmentShader: fragmentShader,
                    side: THREE.DoubleSide,
                    glslVersion: THREE.GLSL3
                });
                
                const planeGeo = new THREE.PlaneGeometry(camsX * 0.1, camsY * 0.1, camsX, camsY);
                const plane = new THREE.Mesh(planeGeo, planeMat);
                
                // Scale to correct aspect ratio and increase display size (doesn't break shader parallax logic)
                const aspect = resX / resY;
                plane.scale.set(aspect * 2.0, 2.0, 1.0);
                
                plane.position.z = -2;
                scene.add(plane);
                
                document.getElementById('placeholder').style.display = 'none';
                if(!reqFrame) render(); // Start animation loop
            } catch (err) {
                logDebug('Error: ' + err.message);
            }
        }
        else if (e.data.type === 'update_sensitivity') {
            if (controls) {
                controls.rotateSpeed = e.data.value;
                controls.panSpeed = e.data.value * 2; // pan is naturally slower
            }
        }
        else if (e.data.type === 'reset_camera') {
            camera.position.set(0, 0, 2);
            camera.up.set(0, 1, 0);
            if (controls) {
                controls.target.set(0, 0, -2);
                controls.update();
            }
        }
        else if (e.data.type === 'update_aperture') {
            if(planeMat) planeMat.uniforms.aperture.value = e.data.value;
        }
    });
  </script>
</body>
</html>
"></iframe>
"""

def load_sample_image(image_path, dataset_name):
    img = Image.open(image_path)
    return img, dataset_name

with gr.Blocks(title="PVSNet/PLFNet", theme="default") as demo:
    gr.Markdown("""
    # PVSNet: Real-Time Position-Aware View Synthesis from Single-View Input
    * Upload a single Lytro image and get a mini parallax video showing capabilities of the light field reconstruction model from our works PVSNet and PLFNet.
    **Note** Huggingface demo is running on CPU, so the inference speed will be slow. It might take around 2 minutes for full LF reconstruction or video generation.
    ### Head to our [Project Page](https://realistic3d-miun.github.io/PVSNet/) for more details about the models.
    """)

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image")
        with gr.Column():
            dataset = gr.Dropdown(
                choices=["Flowers", "Stanford"],
                value="Flowers",
                label="Model Checkpoint (Dataset)"
            )
            resolution = gr.Dropdown(["352x512 (Slow)", "176x256 (Fast)"], value="176x256 (Fast)", label="Resolution")
    
    with gr.Tabs():
        with gr.Tab("Parallax Video"):
            with gr.Row():
                with gr.Column():
                    radius = gr.Slider(0.0006, 0.006, value=0.003, label="Radius")
                    num_frames = gr.Slider(10, 100, value=60, step=10, label="Number of Frames")
                    num_loops = gr.Slider(1, 10, value=4, step=1, label="Number of Loops")
                    generate_vid_btn = gr.Button("Generate Video", variant="primary")
                video_output = gr.Video(label="Generated Video", height=352)

            generate_vid_btn.click(
                fn=process_parallax_video,
                inputs=[img_input, dataset, resolution, radius, num_frames, num_loops],
                outputs=video_output,
            )

        with gr.Tab("Interactive Light Field"):
            with gr.Row():
                with gr.Column():
                    generate_lf_btn = gr.Button("Generate Light Field Data", variant="primary")
                    lf_status = gr.Textbox(label="Status", interactive=False, value="Awaiting generation...")
                    
                    gr.Markdown("### Rendering Controls\nAdjust these parameters and use your mouse to navigate the Light Field (Left Click: Rotate, Right Click: Pan, Scroll: Zoom).")
                    with gr.Row():
                        sensitivity = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Mouse Sensitivity")
                        aperture = gr.Slider(0, 4.5, value=2.2, step=0.1, label="Aperture")
                    reset_btn = gr.Button("Reset Camera")
                
                with gr.Column():
                    gr.HTML(html_code)
            
            # Hidden text box to transfer JSON data to frontend
            b64_frames_state = gr.Textbox(visible=False, elem_id="lf_data_bridge")

            # JS function string to update frames in iframe
            update_js = """(val, a, s) => { 
                if (val) {
                    const frames = JSON.parse(val);
                    const iframe = document.getElementById('lf-iframe');
                    if (iframe && iframe.contentWindow) {
                        iframe.contentWindow.postMessage({
                            type: 'update_frames', 
                            frames: frames,
                            aperture: a,
                            focus: 0,
                            sensitivity: s
                        }, '*');
                    }
                }
            }"""

            # Step 1: Generate Raw Light Field (7x7 array)
            generate_lf_btn.click(
                fn=generate_lf_raw_frames,
                inputs=[img_input, dataset, resolution],
                outputs=[b64_frames_state, lf_status]
            ).then( # Step 2: Run JS to update frontend
                fn=None,
                inputs=[b64_frames_state, aperture, sensitivity],
                outputs=None,
                js=update_js
            )

            # Sliders update iframe instantly via postMessage (no Python execution needed!)
            sensitivity.change(
                fn=None,
                inputs=[sensitivity],
                outputs=None,
                js="(s) => { const iframe = document.getElementById('lf-iframe'); if (iframe && iframe.contentWindow) iframe.contentWindow.postMessage({type: 'update_sensitivity', value: s}, '*'); }"
            )
            
            aperture.change(
                fn=None,
                inputs=[aperture],
                outputs=None,
                js="(a) => { const iframe = document.getElementById('lf-iframe'); if (iframe && iframe.contentWindow) iframe.contentWindow.postMessage({type: 'update_aperture', value: a}, '*'); }"
            )

            reset_btn.click(
                fn=None,
                inputs=None,
                outputs=None,
                js="() => { const iframe = document.getElementById('lf-iframe'); if (iframe && iframe.contentWindow) iframe.contentWindow.postMessage({type: 'reset_camera'}, '*'); }"
            )

    gr.Markdown("### Example Images: Click to Load")
    for dataset_name, images in SAMPLE_IMAGES.items():
        with gr.Accordion(f"📂 {dataset_name} Samples", open=(dataset_name == "Flowers")):
            for row_start in range(0, len(images), 3):
                row_images = images[row_start : row_start + 3]
                with gr.Row():
                    for img_path in row_images:
                        label = os.path.splitext(os.path.basename(img_path))[0]
                        sample_img = gr.Image(
                            img_path,
                            label=label,
                            height=150,
                            interactive=False,
                            show_label=True,
                        )
                        sample_img.select(
                            fn=lambda path=img_path, ds=dataset_name: load_sample_image(path, ds),
                            inputs=[],
                            outputs=[img_input, dataset],
                        )

if __name__ == "__main__":
    demo.launch()