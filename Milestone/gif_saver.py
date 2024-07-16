import glob
import os
import re

from PIL import Image


def create_gif_from_folder(folder_path, output_gif_name='output.gif', duration=500):
    """
    Creates a GIF from all images in the specified folder, sorted by the number X in the filename.

    :param folder_path: Path to the folder containing images
    :param output_gif_name: Name of the output GIF file (default: 'output.gif')
    :param duration: Duration of each frame in milliseconds (default: 500ms)
    """
    # Get list of all image files in the folder
    image_files = glob.glob(os.path.join(folder_path, 'bishop_c1_over_*_of_513914_games.png'))

    if not image_files:
        print(f"No matching image files found in {folder_path}")
        return

    # Function to extract the number X from the filename
    def extract_number(filename):
        match = re.search(r'bishop_c1_over_(\d+)_of_513914_games\.png', filename)
        return int(match.group(1)) if match else 0

    # Sort the files based on the extracted number
    image_files.sort(key=extract_number)

    # Open all images
    images = [Image.open(file) for file in image_files]

    # Save the first image as GIF and append the rest
    images[0].save(
        output_gif_name,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF created successfully: {output_gif_name}")


# Example usage:
create_gif_from_folder('plots/only_log', 'only_log_c1_bishop_dist.gif', 50)
