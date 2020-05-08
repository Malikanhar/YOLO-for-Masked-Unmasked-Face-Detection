import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="fddb annotation file")
    parser.add_argument("output", help="output file")
    args = parser.parse_args()

    with open(args.input, 'r') as file:
        dataset = json.load(file)

    
    annotation = {}

    for ds in tqdm(dataset, 'Processing'):
        annotation[ds['filename']] = []
        for obj in ds['objects']:
            bbox = obj['bbox']
            bbox.append(obj['class'])
            annotation[ds['filename']].append(bbox)

    print("Saving")
    with open(args.output, "w") as file:
        json.dump(annotation, file)

    with open('fddb.categories.txt', 'w') as file:
        file.write('\n'.join(['unmasked', 'masked']))
