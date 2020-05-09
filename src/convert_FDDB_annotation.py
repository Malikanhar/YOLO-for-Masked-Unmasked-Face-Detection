import argparse
import json
import cv2
import re
import os
import numpy as np

def find_dimension(major_radius, minor_radius, angle, cx, cy):
    ux = major_radius * np.cos(angle)
    uy = major_radius * np.sin(angle)
    vx = minor_radius * np.cos(angle + np.pi/2)
    vy = minor_radius * np.sin(angle + np.pi/2)

    w = np.sqrt(ux*ux + vx*vx)
    h = np.sqrt(uy*uy + vy*vy)

    return int(cx - w), int(cy - h), int(w*2), int(h*2)

def main():
    parser = argparse.ArgumentParser(description="Parser for FDDB Annotation Converter")
    parser.add_argument("--annotation", type=str, required=True,
                                    help="path to annotation file,  containing .txt files")
    parser.add_argument("--dataset", type=str, required=True,
                                    help="path to image folder")
    parser.add_argument("--json", type=str, required=True,
                                    help="output annotation .json filename")

    args = parser.parse_args()

    annotation_path = args.annotation
    image_path = args.dataset
    json_filename = args.json

    new_annotation = {}
    err_count = 0

    for file in os.listdir(annotation_path):
        if re.match(r'FDDB-fold-(\d\d)-ellipseList.txt', file):
            with open(os.path.join(annotation_path, file), 'r') as file:
                file_name = file.readline()
                while file_name != "":
                    img = cv2.imread(os.path.join(image_path, file_name[:-1]) + '.jpg')
                    height, width = img.shape[:-1]
                    count = int(file.readline())
                    objects = []
                    for i in range(count):
                        major_radius, minor_radius, angle, cx, cy = file.readline().split(" ")[:-2]
                        major_radius = float(major_radius)
                        minor_radius = float(minor_radius)
                        angle = float(angle)
                        cx = float(cx)
                        cy = float(cy)
                        
                        x, y, w, h = find_dimension(major_radius, minor_radius, angle, cx, cy)
                        
                        if x < 0:
                            print("X : {} should be larger than 0 in {}".format(x, file_name[:-1]))
                            x = 0
                            err_count += 1
                        elif x > width:
                            print("X : {} should be smaller than {} in {}".format(x, width, file_name[:-1]))
                            x = width
                            err_count += 1
                        if y < 0:
                            print("Y : {} should be larger than 0 in {}".format(y, file_name[:-1]))
                            y = 0
                            err_count += 1
                        elif y > height:
                            print("Y : {} should be smaller than {} in {}".format(x, height, file_name[:-1]))
                            y = height
                            err_count += 1
                        if x + w > width:
                            print("X + W : {} should be smaller than {} in {}".format(x+w, width, file_name[:-1]))
                            w = width - x
                            err_count += 1
                        if y + h > height:
                            print("Y + H : {} should be smaller than {} in {}".format(y+h, height, file_name[:-1]))
                            h = height - y
                            err_count += 1
                        if h == 0 or w == 0:
                            print('skipping bbox with no area')
                            continue

                        x = x / width
                        y = y / height
                        w = w / width
                        h = h / height

                        objects.append([x, y, w, h, 0])
                    new_annotation[file_name[:-1] + '.jpg'] = objects
                    file_name = file.readline()

    print('Total annotations fixed : {}'.format(err_count))
    with open(json_filename, 'w') as f:
        json.dump(new_annotation, f)

if __name__ == "__main__":
    main()