import tensorflow as tf
import keras
from keras.models import load_model
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from model.segmentation_model import load_lungs_model

lungs_model = load_lungs_model()

def cropper(test_img):
    test_img = test_img * 255
    test_img = np.uint8(test_img)

    contours, _ = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    filtered_contours = []
    height, width = test_img.shape[:2]
    min_distance_to_edge = min(height,
                               width) * 0.1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > min_distance_to_edge and y > min_distance_to_edge and x + w < width - min_distance_to_edge and y + h < height - min_distance_to_edge:
            filtered_contours.append(contour)

    areas = [cv2.contourArea(c) for c in filtered_contours]
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[
                      :2]

    contour1, contour2 = sorted_contours

    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    if x1 > x2:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    if x1 == x2 and y1 == y2:
        fused_lungs = cv2.resize(test_img[y1:y1 + max(h1, h2), x1:x1 + max(w1, w2)], (256, 256),
                                 interpolation=cv2.INTER_LANCZOS4)
    else:
        right_lung = cv2.resize(test_img[y1:y1 + h1, x1:x1 + w1], (128, 256), interpolation=cv2.INTER_AREA)
        left_lung = cv2.resize(test_img[y2:y2 + h2, x2:x2 + w2], (128, 256), interpolation=cv2.INTER_AREA)

        fused_lungs = np.concatenate((right_lung, left_lung), axis=1)

    return fused_lungs, (x1, y1, w1, h1), (x2, y2, w2, h2)


def convert_to_uint8(image):
    scaled_image = (image * 255).astype(np.uint8)
    scaled_image = scaled_image.reshape(224, 224)
    return scaled_image


def read_nii(filepath, img_size):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    middle_slice_index = array.shape[2] // 2
    initial_img = array[:, :, middle_slice_index]

    img = cv2.resize(initial_img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    xmax, xmin = img.max(), img.min()
    img = (img - xmin) / (xmax - xmin)

    lung_mask = lungs_model.predict(img.reshape(1, img_size, img_size, 1))
    lung_mask, points1, points2 = cropper(convert_to_uint8(lung_mask))
    if points1[1] == points2[1] and points1[0] == points2[0]:
        high = max(points1[3], points2[3])
        width = max(points1[2], points2[2])
    else:
        if points1[1] > points2[1]:
            high = points1[3]
        else:
            high = points2[3]

        if points1[0] > points2[0]:
            width = points1[2]
        else:
            width = points2[2]

    img = img[min(points1[1], points2[1]):max(points1[1], points2[1]) + high,
          min(points1[0], points2[0]):max(points1[0], points2[0]) + width]

    return img