import tensorflow as tf
import numpy as np
from model.masknet import YoloMaskNet2
import cv2
from argparse import ArgumentParser
from model.utils import draw_outputs, load_classes, load_anchors
import time
import requests


url = "http://192.168.8.101:8080/shot.jpg"

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("classes")
    parser.add_argument("anchor")
    parser.add_argument("weight")
    

    args = parser.parse_args()
    print("Load data")
    classes_names = load_classes(args.classes)
    anchors, masks = load_anchors(args.anchor)
    anchors = anchors/416

    yolo = YoloMaskNet2(anchors, masks, len(classes_names), iou_threshold=0.1)
    yolo.build((1, 416, 416, 3))
    img = np.zeros((1, 416, 416, 3))
    output = yolo(img)
    yolo.load_weights(args.weight)



    print("Start")
    # vid = cv2.VideoCapture("video.mp4")

    while True:
        # _, img = vid.read()

        # if img is None:
        #     continue
        img_res = requests.get(url)
        img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr,-1)

        img = cv2.flip(img, 0)

        # img = cv2.resize(img, (416, 416))

        img_in = tf.expand_dims(img, 0)
        img_in = transform_images(img_in, 320)
        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        img = draw_outputs(img, (boxes, scores, classes, nums), classes_names)
        t2 = time.time()



        img = cv2.putText(img, "FPS: {:.2f}".format(1/(t2-t1)), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        cv2.imshow('screen', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()