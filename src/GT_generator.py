import math

import cv2
import numpy as np


def make_mask(forged_image, source_image, name):

    # Compute absolute pixel-wise difference in the RGB channels
    difference = cv2.absdiff(forged_image, source_image)

    # Convert the difference to grayscale
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Normalize the grayscale image to the range [0, 1]
    normalized_diff = gray_diff / 255.0

    # Apply threshold to create binary mask M
    threshold = 0.15
    _, binary_mask = cv2.threshold(normalized_diff, threshold, 1, cv2.THRESH_BINARY)

    # binary_mask = 1 - binary_mask
    # Save or display the binary mask
    # cv2.imwrite('mask_{}.png'.format(name), binary_mask * 255)  # Multiplying by 255 to save as a binary image

    return binary_mask



def make_similarity_map(mask_img, k):

    H, W = mask_img.shape  # Get the dimensions of the mask image
    h = math.ceil(H / k)  # Height of each patch
    w = math.ceil(W / k)  # Width of each patch

    # List to store the forged probabilities of each patch
    forged_probabilities = []

    # Divide the mask image into k x k patches and compute the probability for each patch
    for i in range(k):
        for j in range(k):
            # Extract patch mi
            patch = mask_img[i * h:min((i + 1) * h, H), j * w:min((j + 1) * w, W)]

            # Calculate the forged probability (average pixel value in the patch)
            forged_prob = np.mean(patch)
            forged_probabilities.append(forged_prob)

    # Now compute second-order supervision, i.e., sij for each pair of patches
    second_order_supervision = np.zeros((k * k, k * k))
    for i in range(k * k):
        for j in range(k * k):
            # Euclidean distance-based supervision
            second_order_supervision[i, j] = 1 - (forged_probabilities[i] - forged_probabilities[j]) ** 2

    return second_order_supervision