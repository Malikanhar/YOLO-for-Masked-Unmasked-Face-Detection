import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import cv2

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("anchor", help="Anchor file")
    parser.add_argument("width", type=int, help="Image width")
    parser.add_argument("height", type=int, help="Image height")
    args = parser.parse_args()

    with open(args.anchor, 'r') as file:
        line = file.readline()
        anchors = [anc.split(',') for anc in line.split(" ")]
    
    img = np.zeros((args.width, args.height, 3))
    shape = img.shape[:2]
    center = (shape[0]/2, shape[1]/2)
    for anc in anchors:
        w, h = int(float(anc[0])/2), int(float(anc[1])/2)
        cx, cy = int(center[0]), int(center[1])
        img = cv2.rectangle(img, (cx - w, cy - h), (cx + w, cy + h), (255, 0, 0), 2)
    
    plt.imshow(img.astype('int32'))
    plt.show()
