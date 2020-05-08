import tensorflow as tf

def transform_image(image, size):
    image = tf.image.resize(image, size)
    image = image / 255
    return image

def transform_output(y, anchors, masks, grid_size=13):
    # y : N, jumlah_objek, [x, y, w, h, c]
    output = []
    current_grid = grid_size

    # 1. Find the correct Anchor Box

    # rubah anchor box jadi tensor
    anchors = tf.cast(anchors, tf.float32)
    # hitung luas anchor box
    anchor_area = anchors[..., 0] * anchors[..., 1]
    
    # duplikasi bbox sebanyak anchor box, buat dicari tau mana anchorbox yg paling tepat
    box_wh = y[..., 2:4]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    # hitung luas bbox
    box_area = box_wh[..., 0] * box_wh[..., 1]
    # hitung intersectionya
    # triknya cari width minimum dan height minimum, itulah area yg ber-intersect
    intersect_area = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    # hitung IoU nya
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    
    # pilih anchor yang memiliki IoU terbesar
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    # gabingin y nya sama anchorboxnya
    # jadinya formatnya N, bbox, [x, y, w, h, c, anchor_idx]
    y = tf.concat([y, anchor_idx], axis=-1)

    # 2. Create tensor output
    for anchor_idxs in masks:
        output_tensor = generate_tensor_output(y, current_grid, anchor_idxs)
        output.append(output_tensor)
        current_grid *= 2
    
    # output : (len(masks), (N, grid, grid, anchor, [x, y, w, h, obj, class]))
    return tuple(output)

# pake tf.function untuk operasi tensor
@tf.function
def generate_tensor_output(y, grid_size, anchor_indexes):
    # y : N, bbox, [x, y, w, h, c, anchor_index]
    N = tf.shape(y)[0]

    # output_tensor = N, grid, grid, anchors, [x, y, w, h, obj, class]
    output_tensor = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_indexes)[0], 6))

    # convert masking anchornya jadi int
    anchor_indexes = tf.cast(anchor_indexes, tf.int32)

    # buat variabel buat nyempen index yang nantinya dipake ngisi data ke output_tensor
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    # nilai yg bakal dimasukin ke output_tensor
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    idx = 0
    for i in tf.range(N): # iterasi sejumlah gambar
        for j in tf.range(tf.shape(y)[1]): # iterasi sejumlah bbox
            # cek bboxnya valid apa tidak dgn mengecek w atau h
            if tf.equal(y[i][j][2], 0): # kalo w == 0, berarti gak valid
                continue
            
            # cari tau anchornya sesuai dgn maskingnya apa tidak
            anchor_eq = tf.equal(anchor_indexes, tf.cast(y[i][j][5], tf.int32))
            if tf.reduce_any(anchor_eq):
                # ambil x, y, w, h nya
                box = y[i][j][0:4]

                # hitung x y tengah
                # x y ditambah setengah w h
                center_xy = box[:2] + (box[2:]/2)

                # ambil masknya
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                # hitung letak posisi gridnya
                grid_xy = tf.cast(center_xy // (1/grid_size), tf.int32)

                # sekarang tinggal catet index tensornya
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                # lalu nilai yg bakal dimasukin ke output_tensor
                # x, y, w, h, obj, class dimana x y w h relative terhadap ukuran gambar
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y[i][j][4]])

                idx += 1
    # terakhir gabungin
    return tf.tensor_scatter_nd_update(output_tensor, indexes.stack(), updates.stack())


# Create a dictionary describing the features.
image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/x': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/y': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/w': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/h': tf.io.VarLenFeature(tf.float32),
    'image/object/class': tf.io.VarLenFeature(tf.int64),
}

def parse_image(example_proto, size):
    x = tf.io.parse_single_example(example_proto, image_feature_description)
    x_train = tf.image.decode_jpeg(x['image/encoded'])
    x_train = transform_image(x_train, size)

    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/x']),
                        tf.sparse.to_dense(x['image/object/bbox/y']),
                        tf.sparse.to_dense(x['image/object/bbox/w']),
                        tf.sparse.to_dense(x['image/object/bbox/h']),
                        tf.cast(tf.sparse.to_dense(x['image/object/class']), tf.float32)
                        ], axis=1)
    paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    
    return x_train, y_train
    

def load_dataset(path, size):
    dataset = tf.data.TFRecordDataset(path)
    return dataset.map(lambda x: parse_image(x, size))
    

    
