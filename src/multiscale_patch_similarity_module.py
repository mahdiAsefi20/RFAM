import torch
import torch.nn.functional as F
import cv2
# import matplotlib.pyplot as plt
import numpy as np

#feature_maps = [(U1_low, A1_low), (U2_low, A2_low), (U1_mid, A1_mid), (U2_mid, A2_mid), (U1_high, A1_high), (U2_high, A2_high)]
class MPSM():
    def __init__(self, k=5):
        self.feature_maps = []
        self.k = k

    def fuse_streams(self):
        U_low = (self.feature_maps[0][0] * self.feature_maps[0][1]) + (self.feature_maps[1][0] * self.feature_maps[1][1])
        U_mid = (self.feature_maps[2][0] * self.feature_maps[2][1]) + (self.feature_maps[3][0] * self.feature_maps[3][1])
        U_high = (self.feature_maps[4][0] * self.feature_maps[4][1]) + (self.feature_maps[5][0] * self.feature_maps[5][1])
        return U_low, U_mid, U_high

    def resize_and_concat(self, U_low, U_mid, U_high):

        output_size = U_high.shape[-1]
        U_low_resized = F.interpolate(U_low, size=(output_size, output_size), mode='bicubic', align_corners=False)
        U_mid_resized = F.interpolate(U_mid, size=(output_size, output_size), mode='bicubic', align_corners=False)

        U_final = torch.cat((U_low_resized, U_mid_resized, U_high), 1)
        return U_final

    def make_patch(self, U):
        batch_size, channels, height, width = U.shape

        # Calculate patch size
        patch_height = (height + self.k - 1) // self.k
        patch_width = (width + self.k - 1) // self.k

        # Pad the input tensor to handle edge cases
        padded_input = F.pad(U, (0, (patch_width * self.k - width), 0, (patch_height * self.k - height)),
                             mode='constant', value=0)

        # Reshape and partition into patches
        patches = padded_input.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, k, k, C, patch_height, patch_width)
        return patches

    def flatten_patches(self, patches):
        batch_size, num_patches, num_patches, channels, patch_height, patch_width = patches.size()
        # print(batch_size, num_patches, channels, patch_height, patch_width)
        # Flatten each patch into a 1D vector
        flattened_patches = patches.view(batch_size, num_patches * num_patches,-1)
        return flattened_patches


    def patch_similarities(self, patches):
        # Normalize patch vectors along the feature dimension
        normalized_patches = F.normalize(patches, p=2, dim=-1)
        # Compute cosine similarity using batch matrix multiplication
        similarity = torch.bmm(normalized_patches, normalized_patches.transpose(1, 2))
        # Scale cosine similarity from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        return similarity

    def similarity_map(self, feature_maps: list):

        self.feature_maps = feature_maps

        u1, u2, u3= self.fuse_streams()

        uf= self.resize_and_concat(u1, u2, u3)

        patch  = self.make_patch(uf)

        patch_flat = self.flatten_patches(patch)

        similarity = self.patch_similarities(patch_flat)

        # batch_size = similarity.shape[0]
        # for idx in range(batch_size):
        #     heatmap = similarity[idx].detach().cpu().numpy()  # (k^2, k^2)
        #     plt.figure()
        #     plt.imshow(heatmap, cmap='viridis')
        #     plt.colorbar()
        #     plt.axis('off')
        #     plt.savefig(f"heatmap_{idx}.png", bbox_inches='tight', dpi=300)
        #     plt.close()  # Save as PNG

        return similarity

# U1_low = torch.rand(3,728, 14, 14)
# A1_low = torch.rand(3,1, 14, 14)
#
# U2_low = torch.rand(3,728, 14, 14)
# A2_low = torch.rand(3,1, 14, 14)
#
# U1_mid = torch.rand(3,728, 14, 14)
# A1_mid = torch.rand(3,1, 14, 14)
#
# U2_mid = torch.rand(3,728, 14, 14)
# A2_mid = torch.rand(3,1, 14, 14)
#
# U1_high = torch.rand(3,2048, 7, 7)
# A1_high = torch.rand(3,1, 7, 7)
#
# U2_high = torch.rand(3,2048, 7, 7)
# A2_high = torch.rand(3,1, 7, 7)
#
# feature_maps = [(U1_low, A1_low), (U2_low, A2_low), (U1_mid, A1_mid), (U2_mid, A2_mid), (U1_high, A1_high), (U2_high, A2_high)]
#
# a = MSPS(feature_maps)
# u1, u2, u3= a.fuse_streams()
#
# print(1, u1.shape, u2.shape, u3.shape)
#
# uf= a.resize_and_concat(u1, u2, u3)
# print(2, uf.shape)
#
# patch  = a.make_patch(uf, 5)
# print(3, patch.shape)
#
# patch_flat = a.flatten_patches(patch)
# print(4, patch_flat.shape)
#
# similar = a.patch_similarities(patch_flat)
# print(6, similar.shape)