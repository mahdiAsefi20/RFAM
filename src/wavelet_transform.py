import numpy as np
import pywt
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def rgb_wavelet_rgb(img_rgb: transforms, wavelet_type='haar', section_to_remove='LL'):
    # Convert tensor image to np image
    to_pil = transforms.ToPILImage()
    img_rgb_pil = to_pil(img_rgb)
    img_rgb_np = np.array(img_rgb_pil)

    # Split channels
    channels = cv2.split(img_rgb_np)
    reconstructed_channels = []

    for ch in channels:
        # Apply 2D wavelet transform
        coeffs2 = pywt.dwt2(ch, wavelet_type)
        LL, (LH, HL, HH) = coeffs2

        # Set LL to zeros
        LL_zeroed = np.zeros_like(LL)

        # Reconstruct channel with zeroed LL
        reconstructed = pywt.idwt2((LL_zeroed, (LH, HL, HH)), wavelet_type)
        
        # Clip values and convert to uint8
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # Resize to original shape (may slightly vary after idwt2)
        reconstructed = cv2.resize(reconstructed, (ch.shape[1], ch.shape[0]))
        reconstructed_channels.append(reconstructed)

    # Make np to tensor transforms
    to_tensor = transforms.ToTensor()

    # Merge back into RGB image
    return to_tensor(cv2.merge(reconstructed_channels))

# Load image (as RGB)
# img = cv2.imread('./Test_FF++/Inputs/All/DF (1).png')
# to_tensor = transforms.ToTensor()  # Change to your image path

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_rgb_tensor = to_tensor(img_rgb)
# # Apply wavelet transform and remove LL
# img_no_ll = rgb_wavelet_rgb(img_rgb_tensor)

# # Show original and result
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img_rgb)
# plt.title("Original RGB")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(img_no_ll)
# plt.title("After Removing LL")
# plt.axis('off')

# plt.tight_layout()
# plt.show()
