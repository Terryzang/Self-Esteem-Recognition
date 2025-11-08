import os
import subprocess
from tqdm import tqdm

# Define the path to the OpenFace executable
openface_exe = "D:/"

# Define input and output folder paths
input_folder = "E:/"
output_folder = "E:/"

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all MP4 video files in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.mp4'):
        input_video = os.path.join(input_folder, filename)
        output_dir = os.path.join(output_folder, os.path.splitext(filename)[0])

        # Create an output subfolder for each video
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Build the command line
        command = [
            openface_exe,
            "-f", input_video,
            "-out_dir", output_dir,
            "-au_static",
        ]

        # Run the command
        subprocess.run(command, check=True)

print("All videos have been processed.")
