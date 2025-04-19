import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_mask(forged_image, source_image, label, name):
    if label == 0:
        return np.zeros(source_image.shape[:2])
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
    patch_h = math.ceil(H / k)  # Height of each patch
    patch_w = math.ceil(W / k)  # Width of each patch

    pad_h = k * patch_h - H
    pad_w = k * patch_w - W
    mask_padded = np.pad(mask_img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Reshape into patches: shape will be (k, patch_h, k, patch_w)
    patches = mask_padded.reshape(k, patch_h, k, patch_w)

    # Compute forged probability for each patch by averaging pixel values (resulting shape: (k, k))
    p_matrix = np.mean(patches, axis=(1, 3))

    # Flatten the patch probabilities to a vector of length k*k
    p_vector = p_matrix.flatten()

    # 6. Compute the second-order supervision matrix.
    # For each pair of patches, compute s_ij = 1 - (p_i - p_j)^2.
    diff = p_vector[:, np.newaxis] - p_vector[np.newaxis, :]
    S = 1 - diff ** 2
    second_order_supervision = S

    # rand = random.randint(0, 100000000)
    # cv2.imwrite(f"{rand}_mask.png", mask_img*255)
    # # Save the heatmap
    # cv2.imwrite(f"{rand}_second_order_supervision.png",S*255)  # Save as PNG

    return second_order_supervision