import os
from PIL import Image

# Define the folders
folders = ["./stacked_MLP/", "./stacked_N/", "./stacked_N+P/","./stacked_N+MLP/","./stacked_N+P+MLP/"]
output_folder = "./stacked/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of image names from one of the folders (assuming all folders have the same image names)
image_names = os.listdir(folders[0])

# Process each image
for image_name in image_names:
    # Ensure the image exists in all folders
    images = []
    for folder in folders:
        image_path = os.path.join(folder, image_name)
        if os.path.exists(image_path):
            images.append(Image.open(image_path))
        else:
            print(f"Image {image_name} not found in folder {folder}. Skipping this image.")
            continue
    
    if len(images) == 5:
        # Create a new blank image (768x256) to hold the combined images
        new_image = Image.new('RGB', (1280, 256))

        # Paste the images side by side
        new_image.paste(images[0], (0, 0))
        new_image.paste(images[1], (256, 0))
        new_image.paste(images[2], (512, 0))
        new_image.paste(images[3], (768, 0))
        new_image.paste(images[4], (1024, 0))

        # Save the combined image to the output folder
        new_image.save(os.path.join(output_folder, image_name))

        print(f"Saved combined image: {image_name}")
    else:
        print(f"Skipping {image_name} as not all images were found in the folders.")

print("Image stacking completed.")
