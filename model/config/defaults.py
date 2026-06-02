# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from yacs.config import CfgNode as CN

_C = CN()
_C.seed = 0
_C.output_dir = "./exp_results"
_C.result_name = "exp_data"

_C.num_pos = 64
_C.num_neg_random = 256
_C.num_neg_hard_pc = 128

_C.n_sample_each =  10000

_C.save_every_epoch = 10
_C.training_epochs = 30
_C.continue_training = False
_C.limit_train_batches = 3500
_C.ckpt_path = None




_C.triplane_resolution = 128
_C.triplane_channels_low = 128
_C.triplane_channels_high = 512
_C.transformer_dim = 512
_C.sdf_n_hidden_layers = 4
_C.recursive_split = False
_C.lr = 1e-3
_C.use_train_dataset = False
_C.name = "test"
_C.dataset = CN()
_C.dataset.supervision_method = ["convex"]
_C.dataset.data_type = "objaverse_convex"
_C.dataset.list_dir = "./configs/data_list/"    
_C.dataset.train_list = "train.list"
_C.dataset.val_list = "val.list"
_C.dataset.data_path = ""
_C.dataset.train_num_workers = 7
_C.dataset.val_num_workers = 2
_C.dataset.train_batch_size = 2
_C.dataset.val_batch_size = 2
_C.dataset.predict_batch_size = 1
_C.dataset.input_path = ""
_C.dataset.pc_num_pts = 50000
_C.dataset.num_neg_pairs = 200000
_C.dataset.num_convex_anchor = 1000
_C.dataset.ray_sample_positive = True
_C.dataset.sdf_clip_val = 0.05
_C.dataset.refine_ratio = 80000
_C.dataset.patch_K = 20000
_C.output_pca = False

_C.loss = CN()
_C.loss.triplet = 1.0
_C.loss.sdf = 1.0
_C.loss.l1 = 0.0
_C.loss.balance_weight = [["sam",1], ["cc_part",1]]

_C.use_pvcnn = False
_C.use_pvcnnonly = False

_C.pvcnn = CN()
_C.pvcnn.point_encoder_type = 'pvcnn'
_C.pvcnn.use_point_scatter = True
_C.pvcnn.z_triplane_channels = 64
_C.pvcnn.z_triplane_resolution = 256
_C.pvcnn.unet_cfg = CN()
_C.pvcnn.unet_cfg.depth = 3
_C.pvcnn.unet_cfg.enabled = True
_C.pvcnn.unet_cfg.rolled = True
_C.pvcnn.unet_cfg.use_3d_aware = True
_C.pvcnn.unet_cfg.start_hidden_channels = 32
_C.pvcnn.unet_cfg.use_initial_conv = False

_C.use_2d_feat = False


_C.decomp = CN()
_C.decomp.interior_samples = 10000
_C.decomp.concavity_samples = 6000
_C.decomp.eps = 0.10
_C.decomp.max_parts = 100
_C.decomp.save_seg = False
