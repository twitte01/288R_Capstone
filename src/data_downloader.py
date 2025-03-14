import os
import gdown
import zipfile

folder_id = "1HbwTacbbtGZP3SvydMA4iiehC5VkRkSc"
destination = "."
os.makedirs(destination, exist_ok=True)

print("Downloading files...\n")
gdown.download_folder(
    id=folder_id,
    output=destination,
    quiet=False
)

zip_files = [
    "./data/audio/_background_noise_.zip",
    "./data/audio/Speech Commands.zip",
    "./data/images/Speech Commands_noise.zip",
    "./data/images/Speech Commands (trimmed).zip"
]


print("Extracting data files...\n")

for zip_file in zip_files:
    # check it exists
    if os.path.exists(zip_file):
        print(f'Extracting {zip_file}...')
        try:
            with zipfile.ZipFile(zip_file, 'r') as _zip:
                _zip.extractall(os.path.dirname(zip_file))
            os.remove(zip_file)  # Delete only if extraction worked
        except zipfile.BadZipFile:
            print(f"Error: Corrupt zip file {zip_file}")
    else:
        print(f'Expected zip file "{zip_file}" does not exist')

print("\nFinished")