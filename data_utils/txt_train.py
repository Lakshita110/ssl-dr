import os

# Directory containing the images
image_dir = "/cluster/tufts/cs152l3dclass/areddy05/IDRID/Images/Training/"
output_file = os.path.join(image_dir, "train.txt")

# Get all image filenames in the directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Write the filenames to train.txt
with open(output_file, "w") as f:
    for image in image_files:
        f.write(image + "\n")

print(f"{len(image_files)} image paths written to {output_file}")