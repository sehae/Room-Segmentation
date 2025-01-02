# from roboflow import Roboflow
#
# rf = Roboflow(api_key="mzKxz12MYlYgSE1rEW0h")
# project = rf.workspace("antstryout").project("room-segmentation-x9htn")
# version = project.version(11)
# dataset = version.download("sam2")

# Script to rename roboflow filenames to something SAM 2.1 compatible.
# Maybe it is possible to remove this step tweaking sam2/sam2/configs/train.yaml.
import os
import re

FOLDER = "./data/train"

for filename in os.listdir(FOLDER):
    # Replace all except last dot with underscore
    new_filename = filename.replace(".", "_", filename.count(".") - 1)
    if not re.search(r"_\d+\.\w+$", new_filename):
        # Add an int to the end of base name
        new_filename = new_filename.replace(".", "_1.")
    os.rename(os.path.join(FOLDER, filename), os.path.join(FOLDER, new_filename))
