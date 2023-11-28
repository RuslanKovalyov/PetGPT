# Author: Ruslan Kovalev
# Generate text file with paths to all .txt files in OpenWebText dataset

import os

path = '/Users/ruslan/Downloads/openwebtext'

# Check if the main path exists
if not os.path.exists(path):
    print("The specified path does not exist.")
    exit()

folders = os.listdir(path)
files_list = []

for folder in folders:
    folder_path = os.path.join(path, folder)

    # Check if it's a valid directory
    if not os.path.isdir(folder_path):
        continue

    try:
        files = os.listdir(folder_path)
        files = [file for file in files if file.endswith('.txt')]
        
        for file in files:
            files_list.append(os.path.join(folder_path, file))
    except Exception as e:
        print(f"Error accessing {folder_path}: {e}")

print(len(files_list))
# save list as txt file in openwebtext folder
with open(os.path.join(path, 'paths.txt'), 'w') as f:
    for file in files_list:
        f.write(file + '\n')
