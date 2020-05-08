import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from .darknet53 import DarknetConv, DarknetBlock, DarknetResidual, DarknetBatchNormalization, Darknet53
from .yolo import YoloBox, YoloConv, YoloNMS, YoloOutput

class MaskNet(Model):
    def __init__(self, name="Masknet"):
        super(MaskNet, self).__init__(name=name)
        # output (256, 256)
        self.dnconv1 = DarknetConv(32, 3)
        # darknet block sebanyak 1 berukuran 64. output (128, 128)
        # Darknet block sebanyak 8 berukuran 256
        self.dnblock3 = DarknetBlock(128, 8)
        # Darknet block sebanyak 8 berukuran 512
        self.dnblock4 = DarknetBlock(128, 8)
        # Darknet block sebanyak 4 berukuran 1024
        self.dnblock5 = DarknetBlock(256, 4)

    def call(self, x):
        x = self.dnconv1(x)
        x = x_36 = self.dnblock3(x)
        x = x_61 = self.dnblock4(x)
        x = self.dnblock5(x)
        # di darknet53, outputnya diambil di layer 36, 61, dan diujung
        return x_36, x_61, x

class YoloMaskNet(Model):
    def __init__(self, anchors, mask, classes, size=None, channels=3, training=False, iou_threshold=0.5, score_threshold=0.5):
        super(YoloMaskNet, self).__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.size = size
        self.channels = channels
        self.training = training
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.masknet = MaskNet()
        self.yoloconv1 = YoloConv(512)
        self.output1 = YoloOutput(512, len(mask[0]), classes)
        self.yoloconv2 = YoloConv(256)
        self.output2 = YoloOutput(256, len(mask[1]), classes)
        self.yoloconv3 = YoloConv(128)
        self.output3 = YoloOutput(128, len(mask[2]), classes)

    
    def call(self, x):
        x_36, x_61, x = self.masknet(x)
        x = self.yoloconv1(x)
        output_1 = self.output1(x)
        x = self.yoloconv2((x, x_61))
        output_2 = self.output2(x)
        x = self.yoloconv3((x, x_36))
        output_3 = self.output3(x)

        if (self.training):
            return output_1, output_2, output_3
        
        boxes_0 = Lambda(lambda x: YoloBox(x, self.anchors[self.mask[0]], self.classes), 
                                    name='yolo_boxes_0')(output_1)
        boxes_1 = Lambda(lambda x: YoloBox(x, self.anchors[self.mask[1]], self.classes), 
                                    name='yolo_boxes_0')(output_2)
        boxes_2 = Lambda(lambda x: YoloBox(x, self.anchors[self.mask[2]], self.classes), 
                                    name='yolo_boxes_0')(output_3)
        
        output = Lambda(lambda x: YoloNMS(x, self.anchors, self.mask, self.classes, self.iou_threshold, self.score_threshold),
                                    name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return output 

class YoloMaskNet2(Model):
    def __init__(self, anchors, mask, classes, size=None, channels=3, training=False, iou_threshold=0.5, score_threshold=0.5):
        super(YoloMaskNet2, self).__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.size = size
        self.channels = channels
        self.training = training
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.darknet53 = Darknet53()
        self.yoloconv1 = YoloConv(512)
        self.output1 = YoloOutput(512, len(mask[0]), classes)
        self.yoloconv2 = YoloConv(256)
        self.output2 = YoloOutput(256, len(mask[1]), classes)
        self.yoloconv3 = YoloConv(128)
        self.output3 = YoloOutput(128, len(mask[2]), classes)

    
    def call(self, x):
        x_36, x_61, x = self.darknet53(x)
        x = self.yoloconv1(x)
        output_1 = self.output1(x)
        x = self.yoloconv2((x, x_61))
        output_2 = self.output2(x)
        x = self.yoloconv3((x, x_36))
        output_3 = self.output3(x)

        if (self.training):
            return output_1, output_2, output_3
        
        boxes_0 = Lambda(lambda x: YoloBox(x, self.anchors[self.mask[0]], self.classes), 
                                    name='yolo_boxes_0')(output_1)
        boxes_1 = Lambda(lambda x: YoloBox(x, self.anchors[self.mask[1]], self.classes), 
                                    name='yolo_boxes_0')(output_2)
        boxes_2 = Lambda(lambda x: YoloBox(x, self.anchors[self.mask[2]], self.classes), 
                                    name='yolo_boxes_0')(output_3)
        
        output = Lambda(lambda x: YoloNMS(x, self.anchors, self.mask, self.classes, self.iou_threshold, self.score_threshold),
                                    name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return output 