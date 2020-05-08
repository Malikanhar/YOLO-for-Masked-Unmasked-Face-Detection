from tensorflow.keras.layers import Input
from model.utils import load_darknet_weights
from model.yolo import YoloV3
from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("weights", help="Darknet Weight")
    parser.add_argument("output", help="Output Weight")

    args = parser.parse_args()

    anchor = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
    mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    yolo = YoloV3(anchor, mask, 80)
    img = np.zeros((1, 416, 416, 3))
    output = yolo(img)
    yolo.summary()
    print('Model created')
    load_darknet_weights(yolo, args.weights, False)
    print('Weights loaded')
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')
    yolo.save_weight(args.output)
    print('Weights saved')



