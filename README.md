# YOLO-for-Masked-Unmasked-Face-Detection
Masked and unmasked face detection using YOLOv3

## Convert Dataset Annotation
First, you need to convert the data annotation to the following json:
```bash
{
    image_path : [[x, y, w, h, c], [x, y, w , h, c]],
    image_path : [[x, y, w, h, c], [x, y, w , h, c], [x, y, w , h, c]]
}
```
where `x y w h` represents the bbox coordinates relative to the width and height of the image and `c` represent the object class. In this case, we set the value for class `c` to be 0 which represents that every face does not wear a mask. Furthermore, this face will be augmented to use a mask and the class will be changed to 1.

If you are using FDDB, you can convert the annotation using the following command:
<pre>
python src/convert_FDDB_annotation.py
  --annotation dataset/FDDB/FDDB-folds
  --dataset dataset/FDDB/originalPics
  --json dataset/FDDB/annotation.json
</pre>

And for the Wider-Face annotation:
<pre>
python src/convert_Wider-Face_annotation.py
  --annotation wider_dataset/wider_face_split/wider_face_train_bbx_gt.txt
  --dataset wider_dataset/WIDER_train/images
  --json wider_dataset/annotation.json
</pre>
Now, you have annotation.json

## Data Augmentation [Wearing a Mask]
Because we only have a dataset for face detection (not masked face detection), we have to do an augmentation to wear a mask randomly on our face dataset.
<pre>
python src/generate_masked_face.py
  --annotation annotation.json
  --dataset dataset_path
  --mask mask
</pre>
where `annotation.json`is the json file created by converting the original annotation to our required annotation json, `dataset_path` is the dataset directory and `mask` is mask-images directory provided in this repository.
The results of our augmentation can be seen in the following image
![Augmented Masked Face](https://github.com/Malikanhar/YOLO-for-Masked-Unmasked-Face-Detection/raw/master/assets/generated_masked_face.PNG)

## Generate Anchor
Anchor is a pair of width and height of an object that can be generated using the k-means algorithm by clustering the training data. YoloV3 uses 9 anchor boxes so that each grid can predict up to 9 objects at a time.
<pre>
python src/get_anchor.py
  --annotation annotation.json
</pre>
where `annotation.json`is the json file created by converting the original annotation to our required annotation json. By default, this command will create `anchor.txt` file with 9 generated anchor box.

## Create TFrecord files
<pre>
python src/preprocess.py
  --annotation annotation.json
  --dataset dataset_path
</pre>
By default, this command will create `train.tfrecord` and `val.tfrecord` files with ratio 70% for training and 30% for validation.
