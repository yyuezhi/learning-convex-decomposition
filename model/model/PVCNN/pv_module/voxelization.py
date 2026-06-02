# https://github.com/mit-han-lab/pvcnn
'''
MIT License

Copyright (c) 2019 MIT HAN Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from . import functional as F

__all__ = ['Voxelization']


def my_voxelization(features, coords, resolution):
    # import ipdb
    # ipdb.set_trace()###############################################
    # features = features.contiguous().float()
    # coords = coords.int().contiguous()
    b, c, _ = features.shape
    result = torch.zeros(b, c + 1, resolution * resolution * resolution, device=features.device, dtype=torch.float)
    r = resolution
    r2 = resolution * resolution
    indices = coords[:, 0] * r2 + coords[:, 1] * r + coords[:, 2]
    indices = indices.unsqueeze(dim=1).expand(-1, result.shape[1], -1)
    features = torch.cat([features, torch.ones(features.shape[0], 1, features.shape[2], device=features.device, dtype=features.dtype)], dim=1)
    out_feature = result.scatter_add_(index=indices.long(), src=features, dim=2)
    # import ipdb
    # ipdb.set_trace()  #############
    cnt = out_feature[:, -1:, :]
    zero_mask = (cnt == 0).float()
    cnt = cnt * (1 - zero_mask) + zero_mask * 1e-5
    vox_feature = out_feature[:, :-1, :] / cnt
    return vox_feature.view(b, c, resolution, resolution, resolution)
    # torch.scatter_reduce(input=result, index=indices, value=features, dim=2, reduce='sum')
    # return result.view(b, c, resolution, resolution, resolution)


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0, scale_pvcnn=False):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps
        self.scale_pvcnn = scale_pvcnn
        assert not normalize

    def forward(self, features, coords):
        with torch.no_grad():
            coords = coords.detach()
            # norm_coords = coords - coords.mean(2, keepdim=True)
            # import ipdb
            # ipdb.set_trace()
            # import ipdb
            # ipdb.set_trace()

            if self.normalize:
                norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
            else:
                if self.scale_pvcnn:
                    # print('Use scale pvcnn')
                    # print('Use scale pvcnn')
                    norm_coords = (coords + 1) / 2.0 # [0, 1]
                else:
                    norm_coords = (norm_coords + 1) / 2.0
            norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
            vox_coords = torch.round(norm_coords)#####.int() ##.to(torch.int32)
        # vox_feat = F.avg_voxelize(features, vox_coords, self.r)
        new_vox_feat = my_voxelization(features, vox_coords, self.r)
        return new_vox_feat, norm_coords
        # import ipdb
        # ipdb.set_trace()
        # return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
