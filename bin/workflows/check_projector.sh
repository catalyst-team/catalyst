#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

pip install -r requirements/requirements.txt --quiet --find-links https://download.pytorch.org/whl/cpu/torch_stable.html --upgrade-strategy only-if-needed

################################  pipeline 01  ################################
TEST_DIR='./logs/tb_check'

python -c """
import os
import csv
import cv2
import numpy as np

num_images = 100
dir = '$TEST_DIR'
images_dir = os.path.join(dir, 'images')

os.makedirs(images_dir)

# images
images = []
meta_info = []
for i in range(num_images):
    random_img = np.random.randn(64, 64, 3)
    img_path = os.path.join(images_dir, 'image_{}.jpeg'.format(i))
    cv2.imwrite(img_path, random_img)
    images.append(img_path)
    meta_info.append(str(i))

# csv file
with open(os.path.join(dir, 'images.csv'), 'w', newline='') as csv_file:
    fields = ['images', 'meta']
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    for img, meta in zip(images, meta_info):
        writer.writerow({'images': img, 'meta': meta})

# embedding
vectors = np.random.randn(len(images), 128)
np.save(os.path.join(dir, 'embeddings.npy'), vectors)
"""

PYTHONPATH=.:${PYTHONPATH} \
    python catalyst/contrib/scripts/project_embeddings.py \
    --in-npy="${TEST_DIR}/embeddings.npy" \
    --in-csv="${TEST_DIR}/images.csv" \
    --out-dir=${TEST_DIR} \
    --img-size=64 \
    --img-col='images' \
    --meta-cols='meta' \
    --img-rootpath='.'

rm -rf ${TEST_DIR}
