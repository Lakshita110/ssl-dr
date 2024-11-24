import csv
import os

# Input file paths
csv_file_train = "/cluster/tufts/cs152l3dclass/areddy05/IDRID/Labels/train.csv"
csv_file_valid = "/cluster/tufts/cs152l3dclass/areddy05/IDRID/Labels/validation.csv"
txt_file = "/cluster/tufts/cs152l3dclass/areddy05/IDRID/Images/Training/train.txt" 
output_label_file = "/cluster/tufts/cs152l3dclass/areddy05/IDRID/Images/Training/image_labels.txt"

# Read the CSV into a dictionary: {image: label}
image_label_map = {}
with open(csv_file_train, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for row in reader:
        image_label_map[row[0][9:]] = int(row[1])  # {image_name: label}

with open(csv_file_valid, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for row in reader:
        image_label_map[row[0][9:]] = int(row[1])  # {image_name: label}

# Read the images from the .txt file and create the label file
with open(txt_file, "r") as txt_f, open(output_label_file, "w") as out_f:
    for line in txt_f:
        image_name = line.strip().replace(".jpeg", "")  # Remove the file extension
        label = image_label_map.get(image_name)
        if label is None:
            print(f"Label not found for {image_name}")
            continue
        out_f.write(f"{label}\n")

print(f"Label file created at {output_label_file}")
