from tkinter.dnd import Tester

import numpy as np
import cv2
import torchvision.transforms as transforms
from scipy.fftpack import dct, idct
from torchvision.transforms.v2.functional import to_tensor


def rgb2dct(tensor_image: transforms):
    """
    Get the input image and do Discrete Cosine Transform
    :param tensor_image: input image
    :return: DCT of input image
    """
    grayscale_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the tensor to PIL Image
    ])

    pil_gray_image = grayscale_transform(tensor_image)  # Use the transform as a function on tensor_image

    # Convert the PIL image to a NumPy array
    numpy_image = np.array(pil_gray_image)

    # IMPORTANT: Convert the image to float32 (required for cv2.dct)
    numpy_image_float = np.float32(numpy_image)  # Convert to float32 for DCT

    # Apply DCT using OpenCV
    dct_channels = [dct(dct(numpy_image_float[:, :, i], axis=0, norm='ortho'), axis=1, norm='ortho') for i in range(3)]
    return np.stack(dct_channels, axis=2)


def high_pass_filter(dct_image, alpha= 0.33):
    """
    filtering out the low frequency information to amplify subtle artifacts
    at high frequencies

    :param dct_image: DCT image
    :param alpha: controls the low frequency component to be filtered out

    :return: DCT image of high frequency
    """
    """Apply high-pass filter by zeroing a triangular region in the DCT image."""
    h, w, _ = dct_image.shape
    mask = np.ones((h, w, 3), dtype=np.float32)  # Start with a mask of ones

    # Create a triangular mask
    for i in range(round(alpha * w)):
        for j in range(round(alpha * w) - i - 1):
                mask[i, j, :] = 0  # Zero out the triangle

    return dct_image * mask


def mid_pass_filter(dct_image, alpha_high=0.33, alpha_low=0.33):
    """
    filtering out the low frequency information to amplify subtle artifacts
    at mid frequencies

    :param dct_image: DCT image
    :param alpha: controls the low frequency component to be filtered out

    :return: DCT image of mid frequency
    """
    """Apply mid-pass filter by zeroing a triangular region in the DCT image."""
    h, w, _ = dct_image.shape
    mask = np.ones((h, w, 3), dtype=np.float32)  # Start with a mask of ones
    # Create a triangular mask for high
    for i in range(round(alpha_high * w)):
        for j in range(round(alpha_high * w) - i - 1):
            mask[i, j, :] = 0  # Zero out the triangle

    # Create a triangular mask for low
    for x in range(round(alpha_low * w) -1, -1, -1):
        for y in range((round(alpha_low * w) - x - 1) - 1, -1, -1):
            mask[w-x-1, h-y-1, :] = 0  # Zero out the triangle


    return dct_image * mask

def high_pass_square_filter(dct_image, alpha= 0.5):
    """
    filtering out the low frequency information to amplify subtle artifacts
    at high frequencies

    :param dct_image: DCT image
    :param alpha: controls the low frequency component to be filtered out

    :return: DCT image of high frequency
    """
    """Apply high-pass filter by zeroing a square region in the DCT image."""
    h, w, _ = dct_image.shape
    mask = np.ones((h, w, 3), dtype=np.float32)  # Start with a mask of ones

    # Create a triangular mask
    for i in range(round(alpha * w)):
        for j in range(round(alpha * h)):
                mask[i, j, :] = 0  # Zero out the triangle

    return dct_image * mask

def dct2rgb(dct_image):

    idct_channels = [idct(idct(dct_image[:, :, i], axis=0, norm='ortho'), axis=1, norm='ortho') for i in range(3)]
    return np.stack(idct_channels, axis=2)




def frequency_aware_cue(image_tensor, alpha=0.33):
    """
    Get the input image tensor after augmentation and do DCT, high pass filtering, Inverse DCT
    :param image_tensor: A RGB image in tensor. H * W * C

    :return: an image in tensor type in shape of H * W * 1
    """
    dct_output = rgb2dct(image_tensor)
    # filter_output = mid_pass_filter(dct_output, alpha, alpha)
    filter_output = high_pass_filter(dct_output, alpha)
    idct_output = dct2rgb(filter_output)
    to_tensor_transform = transforms.ToTensor()
    tensor_idct_output = to_tensor_transform(idct_output)

    return tensor_idct_output

## Test
# to_tensor_transform = transforms.ToTensor()
# a = to_tensor_transform(cv2.imread("/home/mahdi/Documents/Projects/RFAM/test/aa.png"))
# b = frequency_aware_cue(a)
# print(b.shape)
