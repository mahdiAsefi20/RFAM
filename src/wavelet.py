import numpy as np
from matplotlib import pyplot as plt
import cv2
import pywt
from pywt._doc_utils import draw_2d_wp_basis, wavedec2_keys

x = cv2.imread(r"/home/mahdi/Documents/Projects/RFAM/dataset/FF/originals/test/429_404_026.png")
# x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
x = x.astype(np.float32)
shape = x.shape

max_lev = 3       # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots

fig, axes = plt.subplots(6, 12, figsize=[14, 8])
for level in range(max_lev + 1):
    if level == 0:
        # show the original image before decomposition
        axes[0, 0].set_axis_off()
        axes[1, 0].imshow(x, cmap=plt.cm.gray)
        axes[1, 0].set_title('Image')
        axes[1, 0].set_axis_off()
        continue

    # plot subband boundaries of a standard DWT basis
    draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                     label_levels=label_levels)
    axes[0, level].set_title(f'{level} level\ndecomposition')

    # compute the 2D DWT
    r_c = pywt.wavedec2(x[:, :, 0], 'haar', mode='periodization', level=level)
    g_c = pywt.wavedec2(x[:, :, 1], 'haar', mode='periodization', level=level)
    b_c = pywt.wavedec2(x[:, :, 2], 'haar', mode='periodization', level=level)
    # normalize each coefficient array independently for better visibility
    c3 = [r_c, g_c, b_c]
    for c in c3:
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title(f'Coefficients\n({level} level)')
        axes[1, level].set_axis_off()

plt.tight_layout()
plt.show()