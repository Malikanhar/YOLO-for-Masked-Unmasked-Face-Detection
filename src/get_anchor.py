import numpy as np
import os
import json
from argparse import ArgumentParser

def iou(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid[0], centroid[1]
        w,h = x[0], x[1]
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_iou(X, centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(iou(X[i], centroids)) 
    return sum/n

def kmeans(X, centroids):
    k, dim = centroids.shape
    N = X.shape[0]
    old_distances = np.zeros((N, k))
    prev = np.ones(N) * -1
    it = 1
    while True:
        print('Calculate distance')
        distances = np.array([1 - iou(a, centroids) for a in X])
        print('Centroids  ', centroids)
        print(f'iteration {it}: distance = {np.sum(np.abs(old_distances - distances))}')

        # find closest centroids
        current = np.argmin(distances, axis=1)

        # check if we should stop right now
        if(prev == current).all():
            print('Centroids = ', centroids)
            return centroids

        # calculate new centroids by closest X's mean
        centroids_sum = np.zeros((k, dim), np.float)
        for i in range(N):
            centroids_sum[current[i]] += X[i]
        for j in range(k):
            centroids[j] = centroids_sum[j]/(np.sum(current==j))

        prev = current.copy()
        old_distances = distances.copy()
        it += 1

def save(filename, X, centroids):
    centroids = sorted(centroids, key= lambda x: x[0] * x[1])
    avg = avg_iou(X, centroids)
    with open(filename, 'w') as file:
        file.write(' '.join([f'{c[0]},{c[1]}' for c in centroids]))
    print('Accuracy :', avg*100)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="Input file")
    parser.add_argument("output", help="output file")
    parser.add_argument("cluster", type=int, help="Cluster num")
    parser.add_argument("--image_width", type=int, help="Image width", default=416)
    parser.add_argument("--image_height", type=int, help="Image height", default=416)
    args = parser.parse_args()

    print("Reading "+args.input)
    
    with open(args.input, "r") as file:
        input_file = json.load(file)

    print("Parsing file")
    boxes = []
    for ant in input_file:
        for bboxc in input_file[ant]:
            x, y, w, h, c = bboxc
            boxes.append(np.array([w, h]))
    boxes = np.array(boxes) * (args.image_width, args.image_height)
    N = boxes.shape[0]

    print("Generate Centroid")
    centroids = boxes[np.random.choice(N, args.cluster, replace=False)]
    print("Doing kmeans")
    centroids = kmeans(boxes, centroids)
    print("Saving")
    save(args.output, boxes, centroids)



# [[118.33484874 254.73511602]
#  [247.75627374 348.01457873]
#  [499.75480582 417.67800671]
#  [ 17.70587606  21.21301458]
#  [137.57987237  66.678763  ]
#  [ 45.11539986  53.22765177]
#  [197.21641298 149.3543668 ]
#  [438.87078458 185.19538099]
#  [ 68.77272232 130.85670911]]

    
    



