import os
from PIL import Image, ImageDraw, ImageFont

# Define the folders
folders = ["./stacked_MLP/", "./stacked_N/", "./stacked_N+P/", "./stacked_N+MLP/", "./stacked_N+P+MLP/"]
output_folder = "./stacked/"

# Define the captions
captions = ["MLP", "Norm", "Norm+PosEnc", "Norm+MLP", "Norm+PosEnc+MLP"]

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of image names from one of the folders (assuming all folders have the same image names)
image_names = os.listdir(folders[0])

# Set the font and size (you may need to adjust the font path based on your system)
try:
    font = ImageFont.truetype("Arial.ttf", 24)  # Adjust font size if needed
except IOError:
    font = ImageFont.load_default()  # Fallback to default font if custom font not found

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
        # Create a new blank image (1280x350) to hold the combined images and captions
        new_image = Image.new('RGB', (1280, 350), (255, 255, 255))  # White background for caption area

        # Paste the images side by side
        new_image.paste(images[0], (0, 94))    # Paste at (0, 94) to leave space for captions
        new_image.paste(images[1], (256, 94))
        new_image.paste(images[2], (512, 94))
        new_image.paste(images[3], (768, 94))
        new_image.paste(images[4], (1024, 94))

        # Draw the captions above each image
        draw = ImageDraw.Draw(new_image)
        for i, caption in enumerate(captions):
            text_width, text_height = draw.textsize(caption, font=font)
            x_position = i * 256 + (256 - text_width) // 2  # Center text in each 256x256 block
            draw.text((x_position, 40), caption, font=font, fill=(0, 0, 0))  # Draw text at y=40

        # Save the combined image to the output folder
        new_image.save(os.path.join(output_folder, image_name))

        print(f"Saved combined image with captions: {image_name}")
    else:
        print(f"Skipping {image_name} as not all images were found in the folders.")

print("Image stacking with captions completed.")
