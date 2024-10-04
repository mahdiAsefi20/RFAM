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
        transforms.Grayscale(),  # Convert to grayscale
    ])

    pil_gray_image = grayscale_transform(tensor_image)  # Use the transform as a function on tensor_image

    # Convert the PIL image to a NumPy array
    numpy_image = np.array(pil_gray_image)

    # IMPORTANT: Convert the image to float32 (required for cv2.dct)
    numpy_image_float = np.float32(numpy_image)  # Convert to float32 for DCT

    # Apply DCT using OpenCV
    dct_image = dct(dct(numpy_image_float, axis=0, norm='ortho'), axis=1, norm='ortho')

    return dct_image


def high_pass_filter(dct_image, alpha= 0.33):
    """
    filtering out the low frequency information to amplify subtle artifacts
    at high frequencies

    :param dct_image: DCT image
    :param alpha: controls the low frequency component to be filtered out

    :return: DCT image of high frequency
    """
    """Apply high-pass filter by zeroing a triangular region in the DCT image."""
    h, w= dct_image.shape
    mask = np.ones((h, w), dtype=np.float32)  # Start with a mask of ones

    # Create a triangular mask
    for i in range(round(alpha * w)):
        for j in range(round(alpha * w) - i - 1):
                mask[i, j] = 0  # Zero out the triangle
    mask2 = np.ones((h, w), dtype=np.float32) * 255
    mm = mask2 * mask

    return dct_image * mask

def dct2rgb(dct_image):

    idct_image = idct(idct(dct_image, axis=0, norm='ortho'), axis=1, norm='ortho')

    return idct_image



def frequency_aware_cue(image_tensor):
    """
    Get the input image tensor after augmentation and do DCT, high pass filtering, Inverse DCT
    :param image_tensor: A RGB image in tensor. H * W * C

    :return: an image in tensor type in shape of H * W * 1
    """

    dct_output = rgb2dct(image_tensor)
    filter_output = high_pass_filter(dct_output, 0.33)
    idct_output = dct2rgb(filter_output)
    to_tensor_transform = transforms.ToTensor()
    tensor_idct_output = to_tensor_transform(idct_output)

    return tensor_idct_output