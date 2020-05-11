import argparse
import os
import sys
import numpy as np
import cv2
import glob
import math
import json
import face_recognition
from PIL import Image
from scipy.spatial import distance
from tqdm import tqdm

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    print('Annotation saved at ' + filename)

def copy_annotation(annotations):
    new_data = []
    for an in annotations:
        new_data.append(an.copy())
    return new_data

def get_center_bbox(bboxs):
    center_bboxs = []
    for bbox in bboxs:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        cx = x + (w / 2)
        cy = y + (h / 2)
        center_bboxs.append([cx, cy])
    return center_bboxs

def get_corresponding_bbox(center_bboxs, c_mask):
    center_distance = [distance.euclidean(c_mask, center_bbox) for center_bbox in center_bboxs]
    id_bbox = np.argmin(center_distance)
    return id_bbox, center_distance[id_bbox]

def generate_id_wears(size, length):
    id_wears = []
    length = size if size > length else length
    for i in range(length):
        id_wear = np.random.choice(2, size)
        while any([np.allclose(id_wear, id_w) for id_w in id_wears + list(np.zeros(length, int))]):
            id_wear = np.random.choice(2, size)
        id_wears.append(id_wear)
    return id_wears

def get_rotated_mask(mask_path, landmark):
    num_mask = len(glob.glob(mask_path + '/*'))
    id_mask = str(np.random.randint(num_mask))
    mask_img = Image.open(os.path.join(mask_path, '_mask_') + id_mask + '.png')

    (mask_h, mask_w) = (mask_img.height, mask_img.width)
    (mask_cX, mask_cY) = (mask_w // 2 , mask_h // 2)

    left_mask_w = landmark['nose_bridge'][1][0] - landmark['chin'][0][0]
    right_mask_w = landmark['chin'][-1][0] - landmark['nose_bridge'][1][0]
    
    if left_mask_w <= 0: left_mask_w = 1
    if right_mask_w <= 0: right_mask_w = 1

    x_mask = landmark['chin'][2][0]
    y_mask = min(landmark['chin'][3][1], landmark['nose_bridge'][1][1], landmark['chin'][-3][1])

    new_mask_w = left_mask_w + right_mask_w
    new_mask_h = landmark['chin'][8][1] - y_mask
    new_mask_size = (new_mask_w, new_mask_h)

    left = landmark['chin'][0]
    right = landmark['chin'][-1]
    angle = math.asin((right[1] - left[1])/(right[0] - left[0]))

    left_mask = mask_img.crop((0, 0, mask_cX, mask_h))
    left_mask = left_mask.resize((left_mask_w, new_mask_h))

    right_mask = mask_img.crop((mask_cX, 0, mask_w, mask_h))
    right_mask = right_mask.resize((right_mask_w, new_mask_h))

    new_mask_img = Image.new('RGBA', new_mask_size)
    new_mask_img.paste(left_mask, (0, 0), left_mask)
    new_mask_img.paste(right_mask, (left_mask_w, 0), right_mask)

    new_mask_cX, new_mask_cY = new_mask_w // 2, new_mask_h // 2
    M = cv2.getRotationMatrix2D((new_mask_cX, new_mask_cY), math.degrees(-angle), 1)
    rotate_mask = cv2.warpAffine(np.array(new_mask_img), M, new_mask_size)
    rotate_mask = Image.fromarray(rotate_mask)
    return rotate_mask, x_mask, y_mask

def wear_mask(face_img, landmarks, annotations, mask_path, w_prop_thrs, save_no_mask):
    masked_faces = []
    new_annotations = []
    center_bboxs = get_center_bbox(annotations)
    id_wears = generate_id_wears(len(landmarks), 0)
    for id_wear in id_wears:
        new_annotation = copy_annotation(annotations)
        masked_face = Image.new('RGBA', face_img.size)
        masked_face.paste(face_img, (0, 0))
        for i, landmark in enumerate(landmarks):
            if id_wear[i] == 0: continue
            else:
                rotated_mask, x_mask, y_mask = get_rotated_mask(mask_path, landmark)
                coor = (x_mask, y_mask)
                relative_coor = (x_mask + (rotated_mask.width/2)) / masked_face.width, y_mask / masked_face.height
                
                id_bbox, distance_bbox_coor = get_corresponding_bbox(center_bboxs, relative_coor)
                relative_mask_w = (rotated_mask.width / masked_face.width)
                mask_w_prop = relative_mask_w / new_annotation[id_bbox][2]
                if distance_bbox_coor > new_annotation[id_bbox][2] or mask_w_prop < w_prop_thrs:
                    continue
                else:
                    new_annotation[id_bbox][4] = 1
                    masked_face.paste(rotated_mask, coor, rotated_mask)
        new_annotations.append(new_annotation)
        masked_faces.append(np.array(masked_face))
    if save_no_mask:
        no_masked_face = Image.new('RGBA', face_img.size)
        no_masked_face.paste(face_img, (0, 0))
        new_annotations.append(annotations)
        masked_faces.append(np.array(no_masked_face))
    return masked_faces, new_annotations

def main():
    parser = argparse.ArgumentParser(description="Parser for FDDB Annotation Converter")
    parser.add_argument("--annotation", type=str, required=True,
                                    help="input annotation filename with .json extensions")
    parser.add_argument("--dataset", type=str, required=True,
                                    help="path to image folder")
    parser.add_argument("--mask", type=str, required=True,
                                    help="path to mask image folder")
    parser.add_argument("--augmented", type=str, default='augmented',
                                    help="path to save augmented image")
    parser.add_argument("--save-original", type=bool, default=True,
                                    help="if True, the original image with no mask will be saved")
    parser.add_argument("--mask-proportion", type=float, default=0.7,
                                    help="the threshold of proportion between the width of the mask and the face"
                                    "to remove non-sense face landmark generated by face recognition library")

    args = parser.parse_args()

    in_json = args.annotation
    in_img_path = args.dataset
    out_json = 'generated_' + os.path.splitext(in_json)[0] + '.json'
    out_img_path = args.augmented
    mask_path = args.mask
    save_no_mask = args.save_original
    mask_w_prop = args.mask_proportion

    annotations = {}
    
    data = load_json(in_json)

    if not os.path.exists(out_img_path):
        print('Creating ' + out_img_path + ' directory')
        os.mkdir(out_img_path)

    print('Start data augmentation')
    for filename in tqdm(data.keys(), 'Processing'):
        face_img = Image.open(os.path.join(in_img_path, filename))
        landmarks = face_recognition.face_landmarks(np.array(face_img))
        face_masks, mask_annotation = wear_mask(face_img, landmarks, data.get(filename), mask_path, mask_w_prop, save_no_mask)
        for i, face_mask in enumerate(face_masks):
            id_img = len(glob.glob(out_img_path + '/*'))
            filename = os.path.join(out_img_path, str(id_img)) + '.jpg'
            cv2.imwrite(filename, cv2.cvtColor(face_mask, cv2.COLOR_RGB2BGR))
            annotations[filename] = mask_annotation[i]

    print('Augmented images saved at ' + out_img_path)
    save_json(annotations, out_json)

if __name__ == "__main__":
    main()
