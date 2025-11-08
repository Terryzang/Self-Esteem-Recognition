import subprocess
import os
from tqdm import tqdm


def transcribe_wav(wenet_path, wav_path):
    # Save the current working directory
    original_cwd = os.getcwd()

    # Change the working directory to the Wenet folder
    os.chdir(wenet_path)

    # Build the command line for transcription
    command = f"wenet --language chinese \"{wav_path}\""

    # Execute the command and capture the output
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Restore the original working directory
    os.chdir(original_cwd)

    # Return the standard output (transcription result)
    return result.stdout


def batch_transcribe(wenet_path, data_folder_path):
    # Get all WAV files from the data folder
    wav_files = [f for f in os.listdir(data_folder_path) if f.endswith('.wav')]

    # Use tqdm to display a progress bar
    for filename in tqdm(wav_files, desc="Processing", unit="file"):
        wav_path = os.path.join(data_folder_path, filename)
        output = transcribe_wav(wenet_path, wav_path)

        # Create a text file with the same name to store the transcription
        txt_path = os.path.join(data_folder_path, filename.replace('.wav', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(output)


# Path to the Wenet directory
wenet_path = 'c:/wenet'

# Path to the folder containing WAV files
data_folder_path = 'E:/'

# Batch process all audio files
batch_transcribe(wenet_path, data_folder_path)
