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

    new_annotation = {}
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

                if x < 0:
                    print("X = {} should be larger than 0 in {} | fixing..".format(x, file_name[:-1]))
                    x = 0
                    err_count += 1
                elif x > width:
                    print("X = {} should be smaller than {} in {} | skipping..".format(x, width, file_name[:-1]))
                    x = width
                    err_count += 1
                    continue
                if y < 0:
                    print("Y = {} should be larger than 0 in {} | fixing..".format(y, file_name[:-1]))
                    y = 0
                    err_count += 1
                elif y > height:
                    print("Y = {} should be smaller than {} in {} | skipping..".format(x, height, file_name[:-1]))
                    y = height
                    err_count += 1
                    continue
                if x + w > width:
                    print("X + W = {} should be smaller than {} in {} | fixing..".format(x+w, width, file_name[:-1]))
                    w = width - x
                    err_count += 1
                if y + h > height:
                    print("Y + H = {} should be smaller than {} in {} | fixing..".format(y+h, height, file_name[:-1]))
                    h = height - y
                    err_count += 1
                if h == 0 or w == 0:
                    print('skipping bbox with no area with x {} y {} w {} h {} in {}'.format(x, y, w, h, file_name[:-1]))                    
                    continue

                x = x / width
                y = y / height
                w = w / width
                h = h / height

                objects.append([x, y, w, h, 0])
            new_annotation[file_name[:-1]] = objects
            file_name = file.readline()

    with open(json_filename, 'w') as f:
        json.dump(new_annotation, f)

if __name__ == "__main__":
    main()