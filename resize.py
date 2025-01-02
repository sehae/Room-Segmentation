import os
from PIL import Image

def resize_images_in_folder(folder_path, output_folder, size=(1024, 1024)):
    """
    Resize all images in a folder to the specified size and save them to the output folder.

    :param folder_path: Path to the folder containing images.
    :param output_folder: Path to the folder where resized images will be saved.
    :param size: Tuple indicating the desired size (width, height).
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            with Image.open(file_path) as img:
                # Resize the image
                resized_img = img.resize(size, Image.LANCZOS)

                # Save the resized image
                output_path = os.path.join(output_folder, file_name)
                resized_img.save(output_path)

                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Skipping {file_name}: {e}")

# Example Usage
source_folder = "./jpg"  # Path to your source folder in Colab
output_folder = "./resize"  # Path to the output folder in Colab

resize_images_in_folder(source_folder, output_folder)
