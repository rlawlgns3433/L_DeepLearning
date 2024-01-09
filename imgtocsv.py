import cv2
import numpy as np
import csv
import os

folder1_path = 'D:/Signal/train\\0'
folder2_path = 'D:/Signal/train\\1'
csv_file = 'image_data_with_labels.csv'

def process_images_to_csv(image_folder, label):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for filename in os.listdir(image_folder):
            img_path = os.path.join(image_folder, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (188, 1))
            row = np.concatenate(([label], resized_image.flatten()))
            writer.writerow(row)

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['label'] + [f"pixel_{i}" for i in range(188)])

process_images_to_csv(folder1_path, 0)
process_images_to_csv(folder2_path, 1)
