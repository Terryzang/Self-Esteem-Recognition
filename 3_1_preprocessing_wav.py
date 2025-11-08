import os
import subprocess

number = 5

# Define the function
def extract_wav_from_mp4(directory, out_directory):
    # Check if the input directory exists
    if not os.path.exists(directory):
        return "Directory not found."

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            # Build input and output file paths
            mp4_path = os.path.join(directory, filename)
            wav_path = os.path.join(out_directory, filename.replace(".mp4", ".wav"))

            # Use FFmpeg to extract audio and convert it to WAV (PCM 16-bit, 16kHz, mono)
            command = [
                "ffmpeg",
                "-i", mp4_path,          # Input MP4 file
                "-vn",                   # Ignore video stream
                "-acodec", "pcm_s16le",  # Set audio codec to PCM 16-bit
                "-ar", "16000",          # Resample to 16 kHz
                "-ac", "1",              # Convert to mono
                wav_path                 # Output WAV file
            ]

            # Execute the command
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return "Extraction complete."

# Run the function
directory_path = f"E:/Task-{number}"  # Replace with your input directory path
out_directory = f'E:/task{number}'
extract_wav_from_mp4(directory_path, out_directory)
