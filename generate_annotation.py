import os
import math

SOURCE_FILE_PATH = f"flat_data/labels/2.2.8.txt"
DESTINATION_DIRECTORY_PATH = "darkflow/dataset/annotations"

frame_template = """
<?xml version="1.0"?>
<annotation>
    <filename>2.2.8-{}.jpg</filename>
    <folder>/home/ludel/Workspace/IA/RN/human_detection/darkflow/dataset/frames</folder>
    <size>
        <width>640</width>
        <height>360</height>
        <depth>3</depth>
    </size>
    {}
</annotation>
"""[1:-1]

object_template = """
    <object>
        <name>person</name>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
"""[1:-1]  # remove \n at beginning and end.

bounding_box_coordinates_indexes = 1, 2, 3, 4
frame_index = 5
data = {}

with open(SOURCE_FILE_PATH) as f:
    for row in f.readlines():
        row = row.split(" ")

        if row[frame_index] not in data:
            data[row[frame_index]] = []

        data[row[frame_index]].append(
            object_template.format(
                *[math.trunc(int(row[index])/6) for index in bounding_box_coordinates_indexes]
            )
        )
for frame_index, objects in data.items():
    with open(os.path.join(DESTINATION_DIRECTORY_PATH, f'2.2.8.{frame_index}.xml'), "w") as f:
        f.write(frame_template.format(frame_index, "\n".join(objects)))
