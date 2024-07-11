import os

def add_blue_to_image_names(directory):
    # Get the list of files in the directory
    files = os.listdir(directory)
    color = 'yellow'
    back_ground = 'bg3'
    # Iterate through each file in the directory
    for filename in files:
        # Check if the file is an image file (you may need to adjust this condition based on your image file extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Split the filename and extension
            name, ext = os.path.splitext(filename)

            # Add "blue" to the name
            new_name = name + "_" + back_ground + ext
            # Construct the paths for the old and new filenames
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_name}")

# Specify the directory containing the images
image_directory = './images/YELLOW BOTTLE/background3'
# Call the function to add "blue" to the names of existing images
add_blue_to_image_names(image_directory)
