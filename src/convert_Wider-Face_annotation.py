import argparse
import json
import cv2
import os

def main():
    parser = argparse.ArgumentParser(description="Parser for Wider Face Annotation Converter")
    parser.add_argument("--annotation", type=str, required=True,
                                    help="annotation file with .txt extensions")
    parser.add_argument("--dataset", type=str, required=True,
                                    help="path to image folder")
    parser.add_argument("--json", type=str, required=True,
                                    help="output annotation .json filename")

    args = parser.parse_args()

    annotation_file = args.annotation
    image_path = args.dataset
    json_filename = args.json

    new_annotation = []
    err_count = 0

    with open(os.path.join(annotation_file), 'r') as file:
        file_name = file.readline()
        while file_name != "":
            img = cv2.imread(os.path.join(image_path, file_name[:-1]))
            height, width = img.shape[:-1]
            count = int(file.readline())
            if count == 0:
                file.readline()
            objects = []
            for i in range(count):
                x, y, w, h = file.readline().split(" ")[:-7]
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)

                x = x / width
                y = y / height
                w = w / width
                h = h / height

                objects.append({
                    "bbox" : [x, y, w, h],
                    "class" : 0
                })
            new_annotation.append({
                "filename": file_name[:-1],
                "objects": objects
            })
            file_name = file.readline()

    with open(json_filename, 'w') as f:
        json.dump(new_annotation, f)

if __name__ == "__main__":
    main()