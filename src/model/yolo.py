import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Concatenate, Lambda, Input, Layer
from tensorflow.keras.losses import Loss, binary_crossentropy, sparse_categorical_crossentropy
from .darknet53 import DarknetConv, Darknet53

class YoloConv(Model):
    def __init__(self, filters):
        super(YoloConv, self).__init__()
        self.dnconv1 = DarknetConv(filters, 1)
        self.dnconv2 = DarknetConv(filters, 1)
        self.dnconv3 = DarknetConv(filters * 2, 3)
        self.dnconv4 = DarknetConv(filters, 1)
        self.dnconv5 = DarknetConv(filters * 2, 3)
        self.dnconv6 = DarknetConv(filters, 1)
        self.upsampling = UpSampling2D(2)
    
    def call(self, x):
        if isinstance(x, tuple):
            x, x_skip = x[0], x[1]
            x = self.dnconv1(x)
            x = self.upsampling(x)
            x = Concatenate()([x, x_skip])
        
        x = self.dnconv2(x)
        x = self.dnconv3(x)
        x = self.dnconv4(x)
        x = self.dnconv5(x)
        x = self.dnconv6(x)
        
        return x

class YoloOutput(Model):
    def __init__(self, filters, anchors, classes):
        super(YoloOutput, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.dnconv1 = DarknetConv(filters * 2, 3)
        self.dnconv2 = DarknetConv(self.anchors * (self.classes + 5), 1, batch_norm=False)

    def call(self, x):
        x = self.dnconv1(x)
        x = self.dnconv2(x)
        # N, grid, grid, anchors, class + 5
        x = Lambda(lambda  x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], 
                                    self.anchors, self.classes + 5)))(x)
        return x

def YoloBox(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes)) relative terhadap cell
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, obj, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1 # yang terakhir
    )

    # di paper page 1 bagian 2.1 Bounding Box Prediction
    box_xy = tf.sigmoid(box_xy)
    obj = tf.sigmoid(obj)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1) # x y w h asli, pake ngitung loss

    # bikin grid sejumlah grid_size (grid_size x grid_size)
    # | 1 2 3 4 ... grid |
    # | 1 2 3 4 ... grid |
    # | 1 2 3 4 ... grid |
    # | 1 2 3 4 ... grid |
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2) # [grid, grid, 1, 2]

    # tambah index grid cell ke box_xy
    # e.g. grid (5, 3) + (0.3, 0.2) = (5.3, 3.2) ==> cell ke (5, 3) pada 
    # posisi 0.3 dan 0.2 di cell tersebut dalam range [0, 1]
    # jadi didapet sigmoid(tx) + c
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    # pw * exp(tw)
    box_wh = tf.exp(box_wh) * anchors

    # cari koordinat kiri atas
    box_xy = box_xy - box_wh / 2
    box_xy2 = box_xy
    bbox = tf.concat([box_xy, box_wh], axis=-1)

    return bbox, obj, class_probs, pred_box

def YoloNMS(outputs, anchors, masks, classes, iou_threshold=0.5, score_threshold=0.5):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        xy = o[0][..., 0:2]
        wh = o[0][..., 2:4] + xy
        xywh = tf.concat([xy, wh], axis=-1)
        b.append(tf.reshape(xywh, (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    return boxes, scores, classes, valid_detections


def CalculateIOU(pred_box, true_box):
    # pred_box: (N, grid, grid, anchor, (x, y, w, h)) relative terhadap image size
    # true_box: (N_obj, (x, y, w, h)) relative terhadap image size
    
    # samain dulu bentuknya jadi (N, grid, grid, anchor, N_obj (x, y, w, h))
    # pred_box: (N, grid, grid, anchor, 1, (x, y, w, h))
    pred_box = tf.expand_dims(pred_box, axis=-2)
    # true_box: (1, N_obj, (x, y, w, h))
    true_box = tf.expand_dims(true_box, axis=0)
    # tinggal samain pake tf.broadcast
    # new_shape: (N, grid, grid, anchor, N_obj, (x, y, w, h))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(pred_box), tf.shape(true_box))
    pred_box = tf.broadcast_to(pred_box, new_shape)
    true_box = tf.broadcast_to(true_box, new_shape)

    # selanjutnya hitung intersection areanya
    intersect_width = tf.maximum(tf.minimum(pred_box[..., 0] + pred_box[..., 2], true_box[..., 0] + true_box[..., 2]) -
                tf.maximum(pred_box[..., 0], true_box[..., 0]), 0)
    intersect_heigh = tf.maximum(tf.minimum(pred_box[..., 1] + pred_box[..., 3], true_box[..., 1] + true_box[..., 3]) -
                tf.maximum(pred_box[..., 1], true_box[..., 1]), 0)
    intersect_area = intersect_width * intersect_heigh

    # hitung luas tiap box
    pred_box_area = pred_box[..., 2] * pred_box[..., 3]
    true_box_area = true_box[..., 2] * true_box[..., 3]
    # hitung IoU
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area)
    # iou: (N, grid, grid, anchors, N_obj)
    return iou


class YoloV3(Model):
    def __init__(self, anchors, mask, classes, size=None, channels=3, training=False):
        super(YoloV3, self).__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.size = size
        self.channels = channels
        self.training = training
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
        
        output = Lambda(lambda x: YoloNMS(x, self.anchors, self.mask, self.classes),
                                    name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return output




class YoloLoss(Loss):
    def __init__(self, masked_anchors, class_num, class_loss='categorical', threshold=0.5, bbox_loss_weight=2):
        super(YoloLoss, self).__init__()
        self.masked_anchors = masked_anchors
        self.class_num = class_num
        self.class_loss = class_loss
        self.threshold = threshold
        self.bbox_loss_weight = bbox_loss_weight

    def call(self, y_true, y_pred):
        # 1. tranform predict output
        # y_pred: (N, grid, grid, masked_anchors, (x, y, w, h, obj, ...class)) dimana x y w h relative terhadap cell size
        pred_box, pred_obj, pred_class, pred_xywh = YoloBox(y_pred, self.masked_anchors, self.class_num)

        # ambil x y
        pred_xy = pred_xywh[..., 0:2]
        # ambil w h
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform true output
        # y_true : (N, grid, grid, anchors, (x, y, w, h, obj, class)) relative terhadap image size
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)

        # ambil w h nya
        true_wh = true_box[..., 2:4]
        # cari titik tengah x, y
        true_xy = true_box[..., 0:2] + (true_wh/2)
        
        # kasi bobot loss, box yg lebih kecil bakal dapet loss yg lebih gede
        box_loss = self.bbox_loss_weight - true_wh[..., 0] * true_wh[..., 1] # luas bbox

        # 3. rubah x y w h yang relative terhadap image size jadi relative terhadap cell size
        # buat grid, biar tau x y nya ada di grid mana
        grid_size = tf.shape(y_true)[1]
        # ini bakal negbikin grid replika
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        # intinya sih ngebikin grid[y][x] == [[x, y]]
        
        # balik persamaan yg ada di YoloBox biar ngerubah yg dari relative image jadi relative cell
        # kalo dipaper dijelasin di halaman 1 section 2.1 Bounding Box Prediction paragraf 2
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)

        # itung w h nya dgn ngebalik persamaan yang dipaper juga biar relative terhadap anchor
        # log nya disini basis e == logaritma natural
        true_wh = tf.math.log(true_wh / self.masked_anchors)
        # tapi ada true_wh yg nilainya 0, yang menyebabkan log(0) == inf
        # untuk mengatasi ini, rubah yg inf jadi 0
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # oke, sekarang kondisi true_xy dan true_wh udah sama degn pred_xy dan pred_wh, sama2 relative terhadap cell size

        # 4. hitung mask buat nentuin hasil prediksi bbox mana aja yg dipake
        # true_obj : (N, grid, grid, anchors, 1)
        # reshape jadi : (N, grid, grid, anchors)
        obj_mask = tf.squeeze(true_obj, axis=-1)
        # ambil bbox yg ada objeknya
        # true_box_flat: (N_obj, (x, y, w, h))
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        # hitung iou antara bbox hasil prediksi dengan bbox sebenarnya, lalu cari IoU yang paling gede tiap cell nya
        # best_iou: (N, grid, grid, anchors)
        best_iou = tf.reduce_max(CalculateIOU(pred_box, true_box_flat), axis=-1)
        # buat maskingnya, yg dibawah threshold = 0, yg diatas threshold = 1
        ignore_mask = tf.cast(best_iou < self.threshold, tf.float32)

        # 5. Hitung Lossnya
        # hitung loss x y pake binary_crossentropy
        # kali dengan box_loss nya juga, biat makin gede lossnya
        # obj_mask buat nentuin cell mana aja yg perlu dikasi loss, karena gk semua cell ada objeknya
        xy_loss = obj_mask * box_loss * binary_crossentropy(true_xy, pred_xy)

        # hitung loss w h pake MSE
        # kali dengan box_loss, biar makin gede lossnya (simpelnya bbox yg areanya kecil dapet loss yg paling gede ketimbang yg areanya gede)
        wh_loss = obj_mask * box_loss * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        # hitung loss objectness pake binary_crossentropy
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        # kali masknya
        # lalu yang objeknya gk ada di grid cell, tapi mungkin disebelahnya yang punya IoU > threshold, kita anggep ada objeknya
        # tapi lossnya = 1 * obj_loss
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        # hitung loss class pake sparse_categorical_crossentropy kalo categorical
        # karena datanya engga dalam one hot encoding
        if self.class_loss == 'categorical':
            class_loss = sparse_categorical_crossentropy(true_class_idx, pred_class)
        elif self.class_loss == 'sparse_binary':
            class_loss = binary_crossentropy(tf.squeeze(tf.one_hot(tf.cast(true_class_idx, tf.uint8), 2, axis=-1), axis=-2) , pred_class)
        else:
            class_loss = binary_crossentropy(true_class_idx, pred_class)
        # jangan lupa dimasking sesuai dgn cell yg ada objeknya
        class_loss = obj_mask * class_loss


        # 6. Jumlahin semua lossnya agar output akhir (N, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss


        

