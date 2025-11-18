from tkinter.dnd import Tester
from wavelet_transform import rgb_wavelet_rgb
import numpy as np
import cv2
import torchvision.transforms as transforms
from scipy.fftpack import dct, idct
from torchvision.transforms.v2.functional import to_tensor
from PIL import Image

def rgb2dct(tensor_image):

    print(tensor_image.min(), tensor_image.max())
    # Convert CxHxW tensor -> HxWxC uint8
    img = tensor_image.permute(1, 2, 0).cpu().numpy()
    img = img.astype(np.float32)

    # Apply 2D DCT per channel
    dct_channels = [
        dct(dct(img[:, :, c], axis=0, norm='ortho'), axis=1, norm='ortho')
        for c in range(3)
    ]
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
    # Apply 2D IDCT per channel
    idct_channels = [
        idct(idct(dct_image[:, :, c], axis=0, norm='ortho'), axis=1, norm='ortho')
        for c in range(3)
    ]
    img = np.stack(idct_channels, axis=2)

    # Clip to valid range
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite("idct.png", img)
    return img




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
    pil_idct = Image.fromarray(idct_output)
    # tensor_iwavelet_output = rgb_wavelet_rgb(image_tensor)

    return pil_idct

## Test
# to_tensor_transform = transforms.ToTensor()
# a = to_tensor_transform(cv2.imread("/home/mahdi/Documents/Projects/RFAM/test/aa.png"))
# b = frequency_aware_cue(a)
# print(b.shape)
