import os

import kagglehub

# Download latest version
download_dir = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
print("Dataset files downloaded to: ", download_dir)

# Move the dataset to this directory
curr_dir = os.path.dirname(os.path.realpath(__file__))
print("Moving files to current directory: ", curr_dir)

# move everything in the returned path to this directory
for file in os.listdir(download_dir):
    old_file = download_dir + "/" + file
    new_file = curr_dir + "/" + file
    os.rename(old_file, new_file)

# remove the empty directory
os.rmdir(download_dir)
