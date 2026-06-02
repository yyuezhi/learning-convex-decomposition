# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import lightning.pytorch as pl
from torch.optim import Adam
from model.utils import  sample_interior_points_grid
from torch.optim.lr_scheduler import CosineAnnealingLR
from .dataloader import InferenceData, ObjaverseConvex
from torch.utils.data import DataLoader
from model.model.triplane import TriplaneTransformer, get_grid_coord 
from model.model.model_utils import VanillaMLP
import torch.nn.functional as F
import torch.nn as nn
import os
import trimesh
import numpy as np
import torch.distributed as dist
from model.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
import gc
import time
from model.utils import *
import networkx as nx
import matplotlib.pyplot as plt
import colorsys, time, os, numpy as np, trimesh, networkx as nx, matplotlib.pyplot as plt
from model.decompose import convex_decomposition

EPS = 1e-4

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,
            transformer_dim=cfg.transformer_dim,   
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=cfg.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(input_dim=64,
                                      output_dim=1, 
                                      out_activation="tanh", 
                                      n_neurons=64, #64
                                      n_hidden_layers=cfg.sdf_n_hidden_layers) #6
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device="cuda",
                shape_min=-1, 
                shape_length=2,
                use_2d_feat=self.use_2d_feat)
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        if self.cfg.dataset.data_type == "synthenic":
            self.grid_coord = get_grid_coord(128)
        else:
            self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')


    def configure_optimizers(self):
        params = [{'params': self.sdf_decoder.parameters(), 'lr': self.cfg.lr * 10},
                  {'params': self.logit_scale, 'lr': self.cfg.lr * 100},]
        if self.use_pvcnn:
            params += [{'params': self.pvcnn.parameters(), 'lr': self.cfg.lr},
                       {'params': self.triplane_transformer.parameters(), 'lr': self.cfg.lr}]

        optimizer = Adam(params)
        lr_scheduler = CosineAnnealingLR(optimizer, 10000, eta_min=0)
        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        if self.cfg.dataset.data_type == "objaverse_convex":
            dataset = ObjaverseConvex(self.cfg, is_train=True)
        else:
            raise ValueError(f"Invalid data type: {self.cfg.dataset.data_type}")
        dataloader = DataLoader(dataset, 
                                num_workers=self.cfg.dataset.train_num_workers,
                                batch_size=self.cfg.dataset.train_batch_size,
                                shuffle=True, 
                                pin_memory=True,
                                drop_last=False)
        return  dataloader
        
    def calc_sdf_loss(self, sdf, mask, planes):
        N = planes.shape[0]
        mask = mask.reshape(N, -1)
        coord = self.grid_coord.unsqueeze(0).repeat(N, 1, 1).cuda()[mask].reshape(N, -1, 3) # N, M, 3
        coord_feat = sample_triplane_feat(planes, coord) # N, M, C
        sdf_pred = self.sdf_decoder(coord_feat) # N, M, 1
        sdf_target = sdf.reshape(N, -1, 1)[mask].reshape(N, -1, 1) #[:, mask, :] # N, M, 1
        sdf_loss = self.mse_loss(sdf_pred, sdf_target).mean()
        return sdf_loss



    def calc_triplet_loss_hardneg_raw_points(
        self,
        planes,
        anchor_points_b,   # (B,N,3)
        pos_points_b,      # (B,N,D,3) padded with -1
        neg_points_b       # (B,N,K,3) padded with -1
    ):
        """
        Batched miner+loss from V1 padded tensors.

        For each batch item b and EACH anchor i, emit exactly ONE triplet row:
        - Positive B:
            * Else: uniform over valid positives of anchor i
        - Hard negatives: multinomial w/ replacement using probs ∝ 1/(||B - neg|| + EPS)
        - Random negatives: uniform w/ replacement
        - If an anchor has no pos or no neg: pad that anchor's row with zeros and mark invalid

        Returns: loss, acc, acc_closest, l1_reg, result_dict
        """
        EPS          = 1e-8
        device       = anchor_points_b.device
        B, N         = anchor_points_b.shape[:2]
        num_pos      = int(getattr(self.cfg, "num_pos", 1))           # kept for compatibility; we still emit 1 per anchor
        num_neg_pc   = int(getattr(self.cfg, "num_neg_hard_pc", 0))
        num_neg_rand = int(getattr(self.cfg, "num_neg_random", 0))
        num_neg_tot  = num_neg_pc + num_neg_rand


        # ---- per-batch triplet mining ----
        PA_all, PB_all, PC_all, valid_masks_all = [], [], [], []
        # print(anchor_points_b.shape, pos_points_b.shape, neg_points_b.shape)
        # exit(0)
        for b in range(B):
            anchors = anchor_points_b[b]            # (N,3)
            pos_b   = pos_points_b[b]               # (N,D,3)
            neg_b   = neg_points_b[b]               # (N,K,3)

            # validity masks (exclude -1 padding)
            pos_valid = (pos_b != -1).all(dim=-1)   # (N,D)
            neg_valid = (neg_b != -1).all(dim=-1)   # (N,K)

            PA_list, PB_list, PC_list, mask_list = [], [], [], []

            for i in range(N):
                A        = anchors[i]                           # (3,)
                pos_pool = pos_b[i][pos_valid[i]]              # (Mi,3)
                neg_pool = neg_b[i][neg_valid[i]]              # (Ki,3)

                # if either pool empty → pad ONE row for THIS anchor
                if pos_pool.numel() == 0 or neg_pool.numel() == 0:
                    PA_list.append(A.view(1, 3))
                    PB_list.append(torch.zeros(1, 3, device=device))
                    PC_list.append(torch.zeros(1, num_neg_tot, 3, device=device))
                    mask_list.append(torch.zeros(1, dtype=torch.bool, device=device))
                    continue

                # --------- POSITIVE SELECTION ---------
                # Uniform positive
                j  = torch.randint(0, pos_pool.shape[0], (1,), device=device).item()
                Bp = pos_pool[j]                                        # (3,)

                # --------- NEGATIVE SELECTION ---------
                if num_neg_tot > 0:
                    # hard-pc multinomial by distance FROM POSITIVE (close = harder ⇒ inverse-distance)
                    if num_neg_pc > 0:
                        dists = torch.norm(neg_pool - Bp[None, :], dim=-1)          # (Ki,)
                        probs = 1.0 / (dists + EPS)
                        probs = probs / probs.sum().clamp_min(EPS)
                        idx_C1 = torch.multinomial(probs, num_neg_pc, replacement=True)  # (num_neg_pc,)
                        C1 = neg_pool[idx_C1]                                        # (num_neg_pc,3)
                    else:
                        C1 = neg_pool.new_zeros((0, 3))

                    # random negatives with replacement
                    if num_neg_rand > 0:
                        idx_C3 = torch.randint(0, neg_pool.shape[0], (num_neg_rand,), device=device)
                        C3 = neg_pool[idx_C3]                                        # (num_neg_rand,3)
                    else:
                        C3 = neg_pool.new_zeros((0, 3))

                    C = torch.cat([C1, C3], dim=0)                                   # (num_neg_tot,3)
                else:
                    C = neg_pool.new_zeros((0, 3))

                # Append one triplet row for this anchor
                PA_list.append(A.view(1, 3))
                PB_list.append(Bp.view(1, 3))
                PC_list.append(C.view(1, num_neg_tot, 3))
                mask_list.append(torch.ones(1, dtype=torch.bool, device=device))

            # concat for this batch item
            PA_b = torch.cat(PA_list, dim=0)                  # (N,3)
            PB_b = torch.cat(PB_list, dim=0)                  # (N,3)
            PC_b = torch.cat(PC_list, dim=0)                  # (N,num_neg_tot,3) or (N,0,3)
            mask_b = torch.cat(mask_list, dim=0)              # (N,)

            PA_all.append(PA_b)
            PB_all.append(PB_b)
            PC_all.append(PC_b)
            valid_masks_all.append(mask_b)

        # ---- stack over batch ----
        PA = torch.stack(PA_all, dim=0)                       # (B,N,3)
        PB = torch.stack(PB_all, dim=0)                       # (B,N,3)
        PC = torch.stack(PC_all, dim=0)                       # (B,N,Ntot,3) or (B,N,0,3)
        valid_masks = torch.stack(valid_masks_all, dim=0)     # (B,N)

        # ---- feature sampling ----
        Bsz, M = PA.shape[0], PA.shape[1]
        Ntot = PC.shape[2] if PC.dim() == 4 else 0

        if Ntot > 0:
            coord = torch.cat((PA, PB, PC.view(Bsz, -1, 3)), dim=1)   # (B, 2N+N·Ntot, 3)
        else:
            coord = torch.cat((PA, PB), dim=1)                        # (B, 2N, 3)

        feats  = sample_triplane_feat(planes, coord)                  # (B, 2N(+N·Ntot), C)
        l1_reg = self.l1_loss(feats, torch.zeros_like(feats)).mean()

        Cdim = feats.shape[-1]
        if Ntot > 0:
            featA, featB, featC = torch.split(feats, [M, M, M * Ntot], dim=1)
            featC = featC.view(Bsz, M, Ntot, Cdim)
        else:
            featA, featB = torch.split(feats, [M, M], dim=1)
            featC = feats.new_zeros(Bsz, M, 0, Cdim)

        featA = featA.unsqueeze(-2)                                # (B,N,1,C)
        featB = featB.unsqueeze(-2)                                # (B,N,1,C)

        logit_scale = float(getattr(self, "logit_scale", getattr(self.cfg, "logit_scale", 10.0)))
        cosAB = torch.exp(F.cosine_similarity(featA, featB, dim=-1) * logit_scale)  # (B,N,1)
        cosAC = torch.exp(F.cosine_similarity(featA, featC, dim=-1) * logit_scale)  # (B,N,N)
        cosBC = torch.exp(F.cosine_similarity(featB, featC, dim=-1) * logit_scale)  # (B,N,N)

        # ---- masked loss ----
        numer  = cosAB.squeeze(-1)                                 # (B,N)
        denom1 = numer + (cosAC.sum(-1) if Ntot > 0 else 0)
        denom2 = numer + (cosBC.sum(-1) if Ntot > 0 else 0)
        loss_pair = -0.5 * (
            torch.log((numer / denom1.clamp_min(EPS)).clamp_min(EPS)) +
            torch.log((numer / denom2.clamp_min(EPS)).clamp_min(EPS))
        )                                                          # (B,N)

        valid_float = valid_masks.float()
        loss = (loss_pair * valid_float).sum() / valid_float.sum().clamp(min=1)

        # ---- metrics ----
        if Ntot > 0:
            acc_negwise = torch.logical_and(cosAB > cosAC, cosAB > cosBC).float()  # (B,N,N)
            acc_pair    = acc_negwise.mean(-1)                                     # (B,N)
            closest     = torch.maximum(cosAC.max(-1)[0], cosBC.max(-1)[0])        # (B,N)
        else:
            acc_pair = torch.ones_like(numer)
            closest  = torch.zeros_like(numer)

        acc = (acc_pair * valid_float).sum() / valid_float.sum().clamp(min=1)
        acc_closest_pair = (numer > closest).float()
        acc_closest = (acc_closest_pair * valid_float).sum() / valid_float.sum().clamp(min=1)

        result_dict = {
            "triplet_acc_closest": acc_closest,
            "cosAB": F.cosine_similarity(featA.squeeze(-2), featB.squeeze(-2), dim=-1),  # (B,N)
            "cosAC": F.cosine_similarity(featA, featC, dim=-1),                          # (B,N,N)
            "cosBC": F.cosine_similarity(featB, featC, dim=-1),                          # (B,N,N)
            "valid_mask": valid_masks,                                                   # (B,N)
        }
        return loss, acc, acc_closest, l1_reg, result_dict

    def training_step(self, batch, batch_idx):
        sdf = batch['sdf']

        # ---- PC features → tri-planes ----
        planes = self.pvcnn(batch['pc'], batch['pc'])
        planes = self.triplane_transformer(planes)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        # ---- SDF loss ----
        sdf_loss = self.calc_sdf_loss(sdf, batch['sdf_mask'], sdf_planes)
        device = sdf_loss.device
        triplet_loss_total = torch.zeros((), device=device)
        l1_reg_total       = torch.zeros((), device=device)

        # ---- iterate over requested supervisions ----
        for head in self.cfg.dataset.supervision_method:
            anchor_key = f"triplet_{head}_anchor"
            pos_key = f"triplet_{head}_positive"
            neg_key = f"triplet_{head}_negative"


            t_loss, t_acc, t_acc_closest, l1_reg, _ = self.calc_triplet_loss_hardneg_raw_points(part_planes, batch[anchor_key], batch[pos_key], batch[neg_key])


            weight = 0
            for key, w in self.cfg.loss.balance_weight:
                if head == key:
                    weight = w
                    break

            triplet_loss_total = triplet_loss_total + weight * t_loss
            l1_reg_total       = l1_reg_total + l1_reg


            self.log(f"train/triplet_loss_{head}", (weight * t_loss).detach().item(), prog_bar=False,on_step=True,on_epoch=True)
            self.log(f"train/triplet_acc_{head}",  t_acc.detach().item(),        prog_bar=False,on_step=True,on_epoch=True)
            self.log(f"train/triplet_acc_closest_{head}", t_acc_closest.detach().item(), prog_bar=False,on_step=True,on_epoch=True)


        # ---- final weighted sum (matches val structure) ----
        loss = (
            sdf_loss * self.cfg.loss.sdf
            + triplet_loss_total * self.cfg.loss.triplet
            + l1_reg_total * self.cfg.loss.l1
        )

        # ---- optimize ----
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # ---- summary logs ----
        self.log("train/lr", opt.param_groups[0]["lr"])
        self.log("train/sdf_loss", sdf_loss.detach().item(), prog_bar=True,on_step=True,on_epoch=True)
        self.log("train/triplet_loss", triplet_loss_total.detach().item(), prog_bar=True,on_step=True,on_epoch=True)
        self.log("train/l1_reg", l1_reg_total.detach().item(), prog_bar=True,on_step=True,on_epoch=True)
        self.log("train/loss", loss.detach().item(), prog_bar=True,on_step=True,on_epoch=True)
        self.log("train/logit_scale", self.logit_scale.detach().item())
        self.log("train/current_epoch", self.current_epoch, sync_dist=True)
        self.log("train/global_step",  self.global_step,  sync_dist=True)

        return

    
    def val_dataloader(self):

        if self.cfg.dataset.data_type == "objaverse_convex":
            objaverse_dataset = ObjaverseConvex(self.cfg, is_train=False)
        else:
            raise ValueError(f"Invalid data type: {self.cfg.dataset.data_type}")
        objaverse_dataloader = DataLoader(objaverse_dataset, 
                                num_workers=self.cfg.dataset.val_num_workers,
                                batch_size=self.cfg.dataset.val_batch_size,
                                shuffle=False, 
                                pin_memory=True,
                                drop_last=False)

        return  [ objaverse_dataloader]    




    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sdf = batch['sdf']

        # ---- PC features → tri-planes ----
        pc_feat = self.pvcnn(batch['pc'], batch['pc'])
        planes = self.triplane_transformer(pc_feat)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        # ---- SDF loss ----
        sdf_loss = self.calc_sdf_loss(sdf, batch['sdf_mask'], sdf_planes)
        device = sdf_loss.device
        triplet_loss_total = torch.zeros((), device=device)
        l1_reg_total       = torch.zeros((), device=device)

        # helper: world-mean a 0-D tensor if DDP is active
        def _gather_mean_scalar(x):
            if dist.is_available() and dist.is_initialized():
                gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, x)
                return torch.stack(gathered).mean().detach()
            return x.detach()

        # ---- iterate over any supervision present in the batch ----
        for head in self.cfg.dataset.supervision_method:
            anchor_key = f"triplet_{head}_anchor"
            pos_key = f"triplet_{head}_positive"
            neg_key = f"triplet_{head}_negative"

            t_loss, t_acc, t_acc_closest, l1_reg, _ = self.calc_triplet_loss_hardneg_raw_points(part_planes, batch[anchor_key], batch[pos_key], batch[neg_key])

            weight = 0
            for key, w in self.cfg.loss.balance_weight:
                if head == key:
                    weight = w
                    break

            triplet_loss_total = triplet_loss_total + weight * t_loss
            l1_reg_total       = l1_reg_total + l1_reg

            # logs per head
            acc_mean = _gather_mean_scalar(t_acc)
            self.log(f"val/triplet_acc_{head}", acc_mean.item(), prog_bar=True)
            self.log(f"val/triplet_acc_closest_{head}", t_acc_closest.detach().item(), prog_bar=True)
            self.log(f"val/triplet_loss_{head}", (weight * t_loss).detach().item(), prog_bar=True)

        # ---- final weighted sum ----
        loss = (
            sdf_loss * self.cfg.loss.sdf
            + triplet_loss_total  * self.cfg.loss.triplet
            + l1_reg_total * self.cfg.loss.l1
        )

        # ---- summary logs ----
        self.log("val/sdf_loss", sdf_loss.detach().item(), prog_bar=True)
        self.log("train/l1_reg", l1_reg_total.detach().item(), prog_bar=True)
        self.log("val/loss", loss.detach().item(), prog_bar=True)
        self.log("val/current_epoch", self.current_epoch, sync_dist=True)
        self.log("val/global_step", self.global_step, sync_dist=True)
        return



    def predict_dataloader(self):
        objaverse_dataset = InferenceData(self.cfg)
        objaverse_dataloader = DataLoader(objaverse_dataset, 
                                num_workers=self.cfg.dataset.val_num_workers,
                                batch_size=self.cfg.dataset.predict_batch_size,
                                shuffle=False, 
                                pin_memory=True,
                                drop_last=False)

        return  [objaverse_dataloader]


    def sample_points(self, vertices, faces, n_point_per_face):
        # Generate random barycentric coordinates
        # borrowed from Kaolin https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/mesh/trianglemesh.py#L43
        n_f = faces.shape[0]
        u = torch.sqrt(torch.rand((n_f, n_point_per_face, 1),
                                    device=vertices.device,
                                    dtype=vertices.dtype))
        v = torch.rand((n_f, n_point_per_face, 1),
                        device=vertices.device,
                        dtype=vertices.dtype)
        w0 = 1 - u
        w1 = u * (1 - v)
        w2 = u * v

        face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
        face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
        face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
        points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
        return points

    def sample_and_mean_memory_save_version(self, part_planes, tensor_vertices, n_point_per_face):
        n_sample_each = self.cfg.n_sample_each # we iterate over this to avoid OOM
        n_v = tensor_vertices.shape[1]
        n_sample = n_v // n_sample_each + 1
        all_sample = []
        all_points = []
        for i_sample in range(n_sample):
            sampled_feature = sample_triplane_feat(part_planes, tensor_vertices[:, i_sample * n_sample_each: i_sample * n_sample_each + n_sample_each,])
            assert sampled_feature.shape[1] % n_point_per_face == 0
            sampled_feature = sampled_feature.reshape(1, -1, n_point_per_face, sampled_feature.shape[-1])
            sampled_feature = torch.mean(sampled_feature, axis=-2)
            all_sample.append(sampled_feature)
            all_points.append(tensor_vertices[:, i_sample * n_sample_each: i_sample * n_sample_each + n_sample_each,].reshape(1, -1, n_point_per_face, 3).mean(axis=-2))
        return torch.cat(all_sample, dim=1), torch.cat(all_points, dim=1)

    def _normalize_vertices_to_unit_box(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float32)
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        extent = float((bbmax - bbmin).max())
        if extent < 1e-12:
            return vertices - center
        scale = 2.0 * 0.9 / extent
        return (vertices - center) * scale

    def _extract_face_features_from_mesh(self, mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        n_faces = int(faces.shape[0])
        if n_faces == 0:
            return np.zeros((0, 1), dtype=np.float32)

        verts_norm = self._normalize_vertices_to_unit_box(verts)
        mesh_norm = trimesh.Trimesh(vertices=verts_norm, faces=faces, process=False)
        pc, _ = trimesh.sample.sample_surface(mesh_norm, int(self.cfg.dataset.pc_num_pts))
        pc_tensor = torch.from_numpy(pc).to(device=self.device, dtype=torch.float32).reshape(1, -1, 3)

        pc_feat = self.pvcnn(pc_tensor, pc_tensor)
        planes = self.triplane_transformer(pc_feat)
        _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        n_point_per_face = 1
        vertices_t = torch.from_numpy(verts_norm).to(device=self.device, dtype=torch.float32)
        faces_t = torch.from_numpy(faces).to(device=self.device, dtype=torch.long)
        tensor_vertices = self.sample_points(vertices_t, faces_t, n_point_per_face)
        tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)

        point_feat, _ = self.sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face)
        point_feat = point_feat[0].contiguous().detach().cpu().numpy()
        denom = np.linalg.norm(point_feat, axis=-1, keepdims=True)
        point_feat = point_feat / np.clip(denom, 1e-12, None)
        return point_feat
        
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        # --------- output dir ----------

        save_dir = f"{self.cfg.result_name}"
        os.makedirs(save_dir, exist_ok=True)

        uid = batch["uid"][0]


        decomp_dir = f"{save_dir}/{uid}"
        os.makedirs(decomp_dir, exist_ok=True)

        
        V = batch["vertices"][0].contiguous().cpu().numpy()
        F = batch["faces"][0].contiguous().cpu().numpy()
        input_mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        if "norm_center" in batch:
            norm_center = batch["norm_center"][0].contiguous().cpu().numpy().astype(np.float64, copy=False)
        else:
            norm_center = np.zeros((3,), dtype=np.float64)

        if "norm_scale" in batch:
            norm_scale = float(batch["norm_scale"][0].item())
        else:
            norm_scale = 1.0
        if (not np.isfinite(norm_scale)) or (abs(norm_scale) < 1e-12):
            norm_scale = 1.0
        inv_norm_scale = 1.0 / norm_scale


        interior_samples = self.cfg.decomp.interior_samples
        seed = self.cfg.seed
        n_samples = self.cfg.decomp.concavity_samples
        eps = self.cfg.decomp.eps
        max_parts = self.cfg.decomp.max_parts
        save_seg = bool(getattr(self.cfg.decomp, "save_seg", False))
        # Lightning predict batch sometimes comes with extra dims
        batch["pc"] = batch["pc"][0].reshape(1, -1, 3)
        N = batch["pc"].shape[0]
        assert N == 1

        pc_feat = self.pvcnn(batch["pc"], batch["pc"])
        planes = self.triplane_transformer(pc_feat)
        _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        n_point_per_face = 1
        tensor_vertices = self.sample_points(batch["vertices"][0], batch["faces"][0].long(), n_point_per_face)
        tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)

        point_feat, _ = self.sample_and_mean_memory_save_version(
            part_planes, tensor_vertices, n_point_per_face
        )  # point_feat: (1, n_faces, C)
        point_feat = point_feat[0].contiguous().cpu().numpy()

        if self.cfg.output_pca:
            from sklearn.decomposition import PCA

            data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
            pca = PCA(n_components=3)
            data_reduced = pca.fit_transform(data_scaled)
            data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
            colors_255 = (data_reduced * 255).astype(np.uint8)
            colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
            colored_mesh.export(f"{decomp_dir}/feat_pca_{uid}.ply")

        denom = np.linalg.norm(point_feat, axis=-1, keepdims=True)
        point_feat = point_feat / np.clip(denom, 1e-12, None)

        face_to_patch_precomputed = batch["face_to_patch"][0].contiguous().cpu().numpy()
        pp_row = batch["PP_global_row"][0].contiguous().cpu().numpy()
        pp_col = batch["PP_global_col"][0].contiguous().cpu().numpy()
        pp_data = batch["PP_global_data"][0].contiguous().cpu().numpy()

        shape_raw = batch["PP_global_shape"]
        if isinstance(shape_raw, (list, tuple)):
            if len(shape_raw) == 0:
                raise ValueError("Empty PP_global_shape in predict batch")
            first = shape_raw[0]
            if isinstance(first, torch.Tensor):
                pp_n = int(first.reshape(-1)[0].item())
            else:
                pp_n = int(first)
        elif isinstance(shape_raw, torch.Tensor):
            pp_n = int(shape_raw.reshape(-1)[0].item())
        else:
            pp_n = int(shape_raw)

        PP_global_components = {
            "row": pp_row.astype(np.int64, copy=False),
            "col": pp_col.astype(np.int64, copy=False),
            "data": pp_data,
            "shape": np.int64(pp_n),
        }

        parts = convex_decomposition(
            mesh=input_mesh,
            face_feats=point_feat,
            eps=eps,
            max_parts=max_parts,
            save_dir=decomp_dir,
            seed=seed,
            interior_samples=interior_samples,
            n_samples=n_samples,
            face_to_patch=face_to_patch_precomputed,
            PP_global_components=PP_global_components,
        )

        # Concavity is computed in normalized coordinates; convert it back to original units.
        for p in parts:
            p["concavity"] = float(p["concavity"]) * inv_norm_scale

        # color LUT by part id
        cmap = plt.colormaps.get_cmap("tab20")
        color_lut = {}
        for p in parts:
            pid = int(p["id"])
            rgb = (np.array(cmap(pid % 20)[:3]) * 255).astype(np.uint8)
            color_lut[pid] = np.append(rgb, 255).astype(np.uint8)
        default_rgba = np.array([160, 160, 160, 255], dtype=np.uint8)

        if save_seg:
            face_rgba = np.tile(default_rgba, (F.shape[0], 1))
            for p in parts:
                pid = int(p["id"])
                face_idx = np.asarray(p["faces"], dtype=np.int64)
                if face_idx.size == 0:
                    continue
                valid = face_idx[(face_idx >= 0) & (face_idx < F.shape[0])]
                face_rgba[valid] = color_lut.get(pid, default_rgba)

            cluster_seg_mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
            cluster_seg_mesh.visual.face_colors = face_rgba
            cluster_seg_path = os.path.join(decomp_dir, f"{uid}_cluster_seg.ply")
            cluster_seg_mesh.export(cluster_seg_path)

        # 1) collection of convex hulls (single mesh)
        hull_meshes = []
        for p in parts:
            h = p["hull"]
            if h is None:
                continue
            pid = int(p["id"])
            rgba = np.tile(color_lut.get(pid, default_rgba), (h.faces.shape[0], 1))
            h2 = h.copy()
            h2.vertices = np.asarray(h2.vertices, dtype=np.float64) * inv_norm_scale + norm_center[None, :]
            h2.visual.face_colors = rgba
            hull_meshes.append(h2)

        if len(hull_meshes) > 0:
            all_hulls = trimesh.util.concatenate(hull_meshes)
            hulls_path = os.path.join(decomp_dir, f"{uid}_hull.ply")
            all_hulls.export(hulls_path)
            print(f"Saved merged hulls: {hulls_path}")
        else:
            print("No valid hulls to export (all parts degenerate).")

        print(f"[OK] Decomposition done: {decomp_dir}")
