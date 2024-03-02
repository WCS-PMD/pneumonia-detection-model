import os
import shutil

def organize_files(directory_path, keyword1, keyword2):
    # Create subdirectories if they don't exist
    subdirectory1 = os.path.join(directory_path, "Bacteria")
    subdirectory2 = os.path.join(directory_path, "Virus")

    if not os.path.exists(subdirectory1):
        os.makedirs(subdirectory1)
    if not os.path.exists(subdirectory2):
        os.makedirs(subdirectory2)

    # List all files in the provided directory
    files = os.listdir(directory_path)

    for file in files:
        file_path = os.path.join(directory_path, file)

        # Check if the file name contains the keywords
        if keyword1 in file:
            destination_path = os.path.join(subdirectory1, file)
            shutil.move(file_path, destination_path)
            print(f"Moved {file} to {subdirectory1}")
        elif keyword2 in file:
            destination_path = os.path.join(subdirectory2, file)
            shutil.move(file_path, destination_path)
            print(f"Moved {file} to {subdirectory2}")
if __name__ == "__main__":
    # Replace 'your_directory_path' with the actual path of the directory containing your files
    directory_path = '/Users/zainkhan/zain/sideProjects/Pnemounia Detector/chest_xray/test/PNEUMONIA'
    keyword1 = "bacteria"
    keyword2 = "virus"

    organize_files(directory_path, keyword1, keyword2)