import numpy as np
import tensorflow as tf
import cv2
from PIL import ImageFont, ImageDraw, Image

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv',
    'yolo_output',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]

def broadcast_iou(box_1, box_2):
    # box_1: (N, grid, grid, anchor, (x1, y1, w, h))
    # box_2: (N_obj, (x1, y1, w, h))

    # broadcast boxes
    # box_1: (N, grid, grid, 1, anchor, (x, y, w, h))
    box_1 = tf.expand_dims(box_1, -2)
    # box_2: (1, N_obj, (x, y, w, h))
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (N, grid, grid, anchor, N_obj, (x1, y1, w, h))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)

def set_darknet_conv(layer, wf):
    print('\t\t'+( 'bn' if layer.batch_norm else 'bias'))
            
    filters = layer.conv.filters
    size = layer.conv.kernel_size[0]
    if len(layer.conv.weights) == 0:
        return layer
    in_dim = layer.conv.weights[0].shape[2]

    if not layer.batch_norm:
        conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
    else:
        # darknet [beta, gamma, mean, variance]
        bn_weights = np.fromfile(wf, dtype=np.float32, count=4*filters)
        # tf [gamma, beta, mean, variance]
        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

    # darknet shape (out_dim, in_dim, height, width)
    conv_shape = (filters, in_dim, size, size)
    conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
    # tf shape (height, width, in_dim, out_dim)
    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

    if layer.batch_norm:
        layer.bn.set_weights(bn_weights)
        layer.conv.set_weights([conv_weights])
    else:
        layer.set_weights([conv_weights, conv_bias])
    return layer

def set_darknet_residual(layer, wf):
    layer.dnconv1 = set_darknet_conv(layer.dnconv1, wf)
    layer.dnconv2 = set_darknet_conv(layer.dnconv2, wf)
    return layer

def set_darknet_block(layer, wf):
    layer.dnconv = set_darknet_conv(layer.dnconv, wf)
    for i, l in enumerate(layer.dnconvs):
        layer.dnconvs[i] = set_darknet_residual(l, wf)
    return layer

def load_darknet_weights(model, weights_file, tiny=False):
    wf = open('yolov3.weights', 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    for sub_model in model.layers:
        print(sub_model.name)
        print("==============================")
        for i, layer in enumerate(sub_model.layers):
            print("\t"+layer.name)
            if(layer.name.startswith("darknet_conv")):
                sub_model.layers[i] = set_darknet_conv(layer, wf)
            elif(layer.name.startswith("darknet_block")):
                sub_model.layers[i] = set_darknet_block(layer, wf)
            elif(layer.name.startswith("darknet_residual")):
                sub_model.layers[i] = set_darknet_residual(layer, wf)

def load_classes(class_file):
    classes = []
    with open(class_file, 'r') as file:
        classes = file.read().split("\n")
    return classes

def load_anchors(anchor_file):
    with open(anchor_file, 'r') as file:
        line = file.readline()[:-1]
        raw_anchors = [anc.split(',') for anc in line.split(' ')]
        anchors = [[float(x) for x in anc] for anc in raw_anchors]
        line = file.readline()
        raw_mask = [anc.split(',') for anc in line.split(' ')]
        mask = [[int(x) for x in anc] for anc in raw_mask]
    return np.array(anchors, np.float32), np.array(mask)
    

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    fontpath = "./model/Ubuntu-Medium.ttf"     
    font = ImageFont.truetype(fontpath, 25)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        color = colors[int(classes[i])%len(colors)]
        text = '{}'.format(class_names[int(classes[i])])
        tw, th = draw.textsize(text, font=font)
        if (x1y1[1] - th) < 0:
            t1 = (x1y1[0], x1y1[1])
        else:
            t1 = (x1y1[0], x1y1[1] - th)
        t2 = (t1[0] + tw, t1[1] + th)
        
        draw.rectangle([x1y1, x2y2], outline=color, width=3)
        draw.rectangle([t1, t2], fill=color)
        
        draw.text(t1, text, font=font, fill=(0, 0, 0))




        # img = cv2.rectangle(img, x1y1, x2y2, color, 3)
        # img = cv2.rectangle(img, (x1y1[0]-1, x1y1[1] - 20), (x1y1[0]+(len(text)*15), x1y1[1]), color, -1)

        # img = cv2.putText(img, text, (x1y1[0], x1y1[1]-5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        
        # img = cv2.putText(img, '{} {:.1f}'.format(
        #     class_names[int(classes[i])], objectness[i]),
        #     x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    img = np.array(img_pil)
    return img