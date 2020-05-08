import tensorflow as tf
from argparse import ArgumentParser
import json
from tqdm import tqdm
import random

# source : https://github.com/tensorflow/models/blob/71943914beaa3a0a74c073657193f7e31a3b1b0e/research/object_detection/utils/dataset_util.py#L25
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# source : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
def create_tf_example(example):
    height = example['height']
    width = example['width']
    encoded_image_data = example['encoded']

    xs = example['xs']
    ys = example['ys']

    ws = example['ws']
    hs = example['hs']

    classes = example['classes']

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/object/bbox/x': float_list_feature(xs),
        'image/object/bbox/y': float_list_feature(ys),
        'image/object/bbox/w': float_list_feature(ws),
        'image/object/bbox/h': float_list_feature(hs),
        'image/object/class': int64_list_feature(classes),
    }))
    return tf_example

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("annotation")
    parser.add_argument("anchor")
    parser.add_argument("dataset_path")
    parser.add_argument("--o_train", default="train.tfrecord")
    parser.add_argument("--o_val", default="val.tfrecord")
    parser.add_argument("--validation", type=int, help="Validation amount")

    args = parser.parse_args()

    with open(args.annotation, 'r') as file:
        dataset = json.load(file)
    
    filenames = list(dataset.keys())
    random.shuffle(filenames)

    writer_train = tf.io.TFRecordWriter(args.o_train)
    writer_val = tf.io.TFRecordWriter(args.o_val)
    num = 0
    for img in tqdm(filenames, 'Loading image'):
        file = open(args.dataset_path+img, "rb").read()
        image = tf.image.decode_jpeg(file, channels=3)
        size = tf.shape(image)
        xs = []
        ys = []
        ws = []
        hs = []
        classes = []
        for bbox in dataset[img]:
            xs.append(bbox[0])
            ys.append(bbox[1])
            ws.append(bbox[2])
            hs.append(bbox[3])
            classes.append(bbox[4])
        example = {
            'encoded': file,
            'height': size[0].numpy(),
            'width': size[1].numpy(),
            'xs': xs,
            'ys': ys,
            'ws': ws,
            'hs': hs,
            'classes': classes,
        }
        tf_example = create_tf_example(example)
        
        if num <= args.validation:
            writer_val.write(tf_example.SerializeToString())
        else:
            writer_train.write(tf_example.SerializeToString())
        num += 1
    writer_val.close()
    writer_train.close()