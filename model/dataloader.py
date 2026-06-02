# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os, json, random, time, os.path as osp
import numpy as np, torch, trimesh, h5py, skimage.measure
from trimesh.ray.ray_pyembree import RayMeshIntersector 
from model.utils import refine_large_faces, build_global_patches_numba, sample_interior_points_grid, quad_to_triangle_mesh
from scipy.sparse import coo_matrix, csr_matrix
import numba as nb
import pymeshlab
# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
MESH_EXTENSIONS = (".obj", ".ply", ".stl", ".off", ".glb", ".gltf")


def list_mesh_files(input_dir):
    mesh_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(MESH_EXTENSIONS):
                mesh_files.append(osp.relpath(osp.join(root, filename), input_dir))
    return sorted(mesh_files)


class ObjaverseConvex(torch.utils.data.Dataset):
    """
    Dataset for Objaverse convex decomposition training and inference.
    Returns:
      'sdf', 'uid', 'view_id', 'dataset', 'sdf_mask', 'pc',
      plus (predict mode) 'vertices', 'faces', 'sdf_grid'.
      plus (training) triplet sampling data based on supervision_method.
    """

    def __init__(
        self,
        cfg,
        is_train=True,
        is_predict=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.is_predict = is_predict
        self.sdf_clip_val = cfg.dataset.sdf_clip_val

        # file list
        list_dir = cfg.dataset.list_dir
        if is_train:
            self.data_list = json.load(open(osp.join(list_dir, cfg.dataset.train_list)))
        else:
            self.data_list = json.load(open(osp.join(list_dir, cfg.dataset.val_list)))
        self.data_list = [key["uid"] for key in self.data_list]


        if not is_train:
            self.data_list = self.data_list[:1000]

        split = "train" if is_train else ("predict" if is_predict else "val")
        print(f"{split} dataset len:", len(self.data_list))

        # params
        self.pc_num_pts = cfg.dataset.pc_num_pts
        self.supervision_method = cfg.dataset.supervision_method

        self.rng = np.random.default_rng()

        # paths
        self.data_path = cfg.dataset.data_path
        self.num_neg_pairs = cfg.dataset.num_neg_pairs
        self.num_convex_anchor = cfg.dataset.num_convex_anchor
        self.ray_sample_positive = cfg.dataset.ray_sample_positive

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    def _gen_sdf_mask(self, sdf, tot=1000000, ratio=0.8):
        clip = self.sdf_clip_val
        mask = np.zeros_like(sdf, bool)
        near = np.abs(sdf) < clip
        k_abs = min(int(ratio * tot), near.sum())
        k_uni = tot - k_abs

        idx_abs = np.column_stack(np.where(near))
        sel_abs = idx_abs[np.random.choice(len(idx_abs), k_abs, False)]
        mask[tuple(sel_abs.T)] = True

        rem = np.flatnonzero(~mask.ravel())
        sel_uni = np.random.choice(rem, k_uni, False)
        mask[np.unravel_index(sel_uni, sdf.shape)] = True
        return mask
    def _sample_convex_triplet(
        self,
        input_mesh,
        n_dir_per_anchor=64,
        batch_rays=1_000_000,
        snap_eps=1e-6,
        seed=0,
        neg_pool_size=100_000,
        num_neg_pairs=1024,
        STOP_EPS=1e-4,
        JITTER=0.05
    ):
        """
        Returns:
            anchor_points: (N,3) torch.float32
            pos_points:    (N,n_dir_per_anchor,3) torch.float32, filled; remaining entries = -1
            neg_points:    (N,num_neg_pairs,3)    torch.float32, filled; remaining entries = -1
        """
        rng = np.random.default_rng(seed)

        # -------- Anchors & normals --------
        anchor_np, face_ids = trimesh.sample.sample_surface(input_mesh, self.num_convex_anchor)
        anchor_points = torch.from_numpy(anchor_np.astype(np.float32))          # (N,3)
        N = anchor_points.shape[0]

        nA_all = torch.from_numpy(input_mesh.face_normals[face_ids].astype(np.float32))  # (N,3)
        nA_all = nA_all / (nA_all.norm(dim=1, keepdim=True) + 1e-12)

        # -------- Ray intersector --------
        surf = trimesh.Trimesh(vertices=input_mesh.vertices, faces=input_mesh.faces, process=False)
        rmi = RayMeshIntersector(surf)

        # =======================
        # Common A→B candidate set (used for negatives, and for positives when ray_sample_positive=False)
        # =======================
        pool_np, _ = trimesh.sample.sample_surface(input_mesh, int(neg_pool_size))
        pool_pts = torch.from_numpy(pool_np.astype(np.float32))                 # (P,3)

        K = int(num_neg_pairs)
        total = N * K
        owners = torch.arange(N, dtype=torch.long).repeat_interleave(K)         # (total,)
        cand_idx = torch.from_numpy(rng.integers(0, pool_pts.shape[0], size=(N, K))).long()
        A = anchor_points[owners]                                               # (total,3)
        B = pool_pts[cand_idx.reshape(-1)]                                      # (total,3)
        nA = nA_all[owners]                                                     # (total,3)

        v = B - A
        dist = torch.linalg.norm(v, dim=1)                                      # (total,)
        safe = dist > 1e-12
        u = torch.zeros_like(v)
        u[safe] = v[safe] / dist[safe].unsqueeze(1)

        # Hemisphere gate relative to anchor normal
        hemi_ok = (nA * u).sum(dim=1) < 0.0                                     # (total,)
        already_neg = (~hemi_ok) & safe                                         # outside hemisphere -> NEG

        # Rays needed only for hemi_ok & safe
        need_mask = hemi_ok & safe
        need_idx = need_mask.nonzero(as_tuple=False).squeeze(-1)

        # Start at A + offset*u with offset = min(snap_eps, 0.5*||B-A||)
        offset = torch.minimum(torch.full_like(dist, snap_eps), 0.5 * dist)     # (total,)
        effective_dist = dist - offset                                          # (total,)

        # hit_before[g] == True if first hit occurs strictly before the target B (by STOP_EPS margin)
        hit_before = torch.zeros(total, dtype=torch.bool)

        s = 0
        while s < need_idx.numel():
            e = min(s + batch_rays, need_idx.numel())
            idx = need_idx[s:e]

            org = (A[idx] + offset[idx, None] * u[idx]).numpy()
            dirr = u[idx].numpy()

            loc, index_ray, _ = rmi.intersects_location(org, dirr, False)
            if index_ray.size > 0:
                g = idx[index_ray]                                              # global indices of rays with a hit
                dloc = torch.from_numpy(np.linalg.norm(loc - org[index_ray], axis=1)).to(dist.dtype)
                gap = effective_dist[g] - dloc                                   # >0 => hit before target
                early = gap > STOP_EPS
                if early.any():
                    hit_before[g[early]] = True
            s = e

        # =======================
        # POSITIVES
        # =======================
        pos_points = torch.full((N, n_dir_per_anchor, 3), -1.0, dtype=torch.float32)

        if self.ray_sample_positive:
            # ---------- Original hemisphere raycast positives ----------
            # Fibonacci base directions (uniform on S^2)
            k = np.arange(n_dir_per_anchor, dtype=np.float32) + 0.5
            phi = np.arccos(1 - 2 * (k / n_dir_per_anchor))
            golden = np.pi * (1 + 5**0.5)
            theta = golden * k
            dirs_base = np.stack([np.sin(phi)*np.cos(theta),
                                  np.sin(phi)*np.sin(theta),
                                  np.cos(phi)], axis=1).astype(np.float32)      # (D,3)

            dirs_nd = np.broadcast_to(dirs_base[None, :, :], (N, n_dir_per_anchor, 3)).copy()
            jitter = rng.standard_normal(size=(N, n_dir_per_anchor, 3)).astype(np.float32) * JITTER
            dirs_nd = dirs_nd + jitter
            norms = np.linalg.norm(dirs_nd, axis=2, keepdims=True)
            dirs_nd = dirs_nd / np.maximum(norms, 1e-12)

            # Back hemisphere of nA
            nA_np = nA_all.numpy()                                              # (N,3)
            hemi_mask_np = np.matmul(dirs_nd, nA_np[:, :, None])[:, :, 0] < 0.0 # (N,D) bool
            anc_idx, dir_idx = np.nonzero(hemi_mask_np)

            if anc_idx.size > 0:
                A_all = anchor_points[anc_idx]
                D_all = torch.from_numpy(dirs_nd[anc_idx, dir_idx, :])
                origins = (A_all + snap_eps * D_all).numpy()
                directions = D_all.numpy()

                s = 0
                while s < origins.shape[0]:
                    e = min(s + batch_rays, origins.shape[0])
                    loc, index_ray, _ = rmi.intersects_location(origins[s:e], directions[s:e], False)
                    if loc.shape[0] > 0:
                        global_idx = s + index_ray
                        hit_anc = torch.from_numpy(anc_idx[global_idx]).long()
                        hit_dir = torch.from_numpy(dir_idx[global_idx]).long()
                        loc_t = torch.from_numpy(loc.astype(np.float32))
                        pos_points[hit_anc, hit_dir, :] = loc_t
                    s = e
        else:
            # ---------- Alternative: A→B (same procedure family as negatives) ----------
            # "Positive" here means: within back hemisphere AND NO hit before B (i.e., clear line-of-sight)
            # Build mask over all candidates
            clear_before = torch.zeros(total, dtype=torch.bool)
            clear_before[need_idx] = ~hit_before[need_idx]
            is_pos_alt = hemi_ok & safe & clear_before                           # (total,)

            # Vectorized scatter up to n_dir_per_anchor per owner
            idx = is_pos_alt.nonzero(as_tuple=False).squeeze(-1)                 # (M,)
            if idx.numel() > 0:
                owners_sel = owners[idx]                                         # (M,)
                B_sel = B[idx]                                                   # (M,3)

                order = torch.argsort(owners_sel, stable=True)
                owners_sorted = owners_sel[order]
                B_sorted = B_sel[order]

                change = torch.ones_like(owners_sorted, dtype=torch.bool)
                change[1:] = owners_sorted[1:] != owners_sorted[:-1]
                start_indices = torch.nonzero(change, as_tuple=False).squeeze(-1)
                ends = torch.cat([start_indices[1:], torch.tensor([owners_sorted.numel()],
                                    device=owners_sorted.device)])
                lengths = ends - start_indices
                starts_repeated = torch.repeat_interleave(start_indices, lengths)

                pos_in_group = torch.arange(owners_sorted.numel(),
                                   device=owners_sorted.device) - starts_repeated

                keep_mask = pos_in_group < n_dir_per_anchor
                if keep_mask.any():
                    owners_keep = owners_sorted[keep_mask]
                    slots_keep  = pos_in_group[keep_mask]
                    B_keep      = B_sorted[keep_mask]
                    pos_points[owners_keep, slots_keep, :] = B_keep

        # =======================
        # NEGATIVES (same as before)
        # =======================
        # Negative if: outside hemisphere OR hit occurs before target
        is_neg = (already_neg | hit_before) & safe

        neg_points = torch.full((N, K, 3), -1.0, dtype=torch.float32)
        idx = is_neg.nonzero(as_tuple=False).squeeze(-1)                         # (M,)
        if idx.numel() > 0:
            owners_sel = owners[idx]
            B_sel = B[idx]

            order = torch.argsort(owners_sel, stable=True)
            owners_sorted = owners_sel[order]
            B_sorted = B_sel[order]

            change = torch.ones_like(owners_sorted, dtype=torch.bool)
            change[1:] = owners_sorted[1:] != owners_sorted[:-1]
            start_indices = torch.nonzero(change, as_tuple=False).squeeze(-1)
            ends = torch.cat([start_indices[1:], torch.tensor([owners_sorted.numel()],
                                device=owners_sorted.device)])
            lengths = ends - start_indices
            starts_repeated = torch.repeat_interleave(start_indices, lengths)

            pos_in_group = torch.arange(owners_sorted.numel(),
                               device=owners_sorted.device) - starts_repeated

            keep_mask = pos_in_group < K
            if keep_mask.any():
                owners_keep = owners_sorted[keep_mask]
                slots_keep  = pos_in_group[keep_mask]
                B_keep      = B_sorted[keep_mask]
                neg_points[owners_keep, slots_keep, :] = B_keep

        return anchor_points, pos_points, neg_points

    # -----------------------------------------------------------------
    # main load
    # -----------------------------------------------------------------
    def _load_sample(self, uid):
        if self.is_predict:
            np.random.seed(0)

        # --- load SDF ---
        with h5py.File(osp.join(self.data_path, uid + ".h5"), "r") as hf:
            sdf_np = hf["sdf"][:]

        # --- load mesh and align to SDF MC AABB (kept from your version) ---
        v, f, _, _ = skimage.measure.marching_cubes(sdf_np, 0)
        v = v / 255.0 * 2 - 1



        # --- MC mesh for convex field ---
        mesh_mc = trimesh.Trimesh(v, f, process=False)

        # --- SDF tensor & mask ---
        sdf_mask = self._gen_sdf_mask(sdf_np)
        sdf = torch.clamp(torch.tensor(sdf_np, dtype=torch.float32),
                          -self.sdf_clip_val, self.sdf_clip_val) / self.sdf_clip_val

        sample = {
            "sdf": sdf.unsqueeze(0),
            "uid": uid,
            "view_id": 0,
            "dataset": 0,
            "sdf_mask": torch.tensor(sdf_mask),
        }



        pc_all = mesh_mc.sample(self.pc_num_pts)
        sample["pc"] = torch.tensor(pc_all, dtype=torch.float32)


        if "convex" in self.supervision_method:
            input_mesh = mesh_mc
            anchor_pc, positive_pc, negative_pc = self._sample_convex_triplet(input_mesh, num_neg_pairs= self.num_neg_pairs)
            sample["triplet_convex_anchor"] = anchor_pc
            sample["triplet_convex_positive"] = positive_pc
            sample["triplet_convex_negative"] = negative_pc

        # --- predict extras ---
        if self.is_predict:
            verts2, faces2, _, _ = skimage.measure.marching_cubes(sdf_np, 0)
            verts2 = verts2 / 255.0 * 2 - 1
            sample.update({
                "vertices": torch.from_numpy(verts2.copy()).contiguous(),
                "faces": torch.from_numpy(faces2.copy()).contiguous(),
                "sdf_grid": torch.from_numpy(sdf_np.astype(np.float32))
            })

        return sample

    # -----------------------------------------------------------------
    # torch interface
    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        uid = self.data_list[idx]
        try:
            return self._load_sample(uid)
        except Exception as e:
            print("Error on", uid, ":", e)
            return self.__getitem__(np.random.randint(len(self)))







class InferenceData(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        input_path = cfg.dataset.input_path
        if osp.isfile(input_path):
            self.data_path = osp.dirname(input_path) or "."
            self.data_list = [osp.basename(input_path)]
        elif osp.isdir(input_path):
            self.data_path = input_path
            self.data_list = list_mesh_files(self.data_path)
        else:
            raise FileNotFoundError(f"dataset.input_path not found: {input_path}")

        if len(self.data_list) == 0:
            raise FileNotFoundError(f"No mesh files found in dataset.input_path: {input_path}")

        print("dataset len:", len(self.data_list))

        self.pc_num_pts = cfg.dataset.pc_num_pts
        
        # Patch computation parameters
        self.patch_K = cfg.dataset.patch_K
        self.patch_seed = cfg.seed
        

        self.refine_ratio = cfg.dataset.refine_ratio

    def __len__(self):
        return len(self.data_list)



    def preprocess_mesh(self,mesh):

        ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)

        # Create a MeshSet and add your mesh
        ms = pymeshlab.MeshSet()
        ms.add_mesh(ml_mesh, "from_trimesh")

        # Apply filters
        ms.apply_filter('meshing_remove_duplicate_faces')
        ms.apply_filter('meshing_remove_duplicate_vertices')
        percentageMerge = pymeshlab.PercentageValue(0.1)
        ms.apply_filter('meshing_merge_close_vertices', threshold=percentageMerge)
        ms.apply_filter('meshing_remove_unreferenced_vertices')

        # Save or extract mesh
        processed = ms.current_mesh()
        mesh.vertices = processed.vertex_matrix()
        mesh.faces = processed.face_matrix()


        trimesh.repair.fix_normals(mesh)  # fixes winding + normals to be consistent
        if mesh.is_watertight and mesh.volume < 0:
            mesh.invert()
        return mesh

    def load_mesh(self, file_path):
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Trimesh):
            pass
        elif isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        else:
            raise ValueError(f"Unknown mesh type: {type(mesh)}")
        return mesh

    def normalize_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        extent = float((bbmax - bbmin).max())
        if extent < 1e-12:
            scale = 1.0
            vertices_norm = vertices - center
        else:
            scale = 2.0 * 0.9 / extent
            vertices_norm = (vertices - center) * scale
        mesh.vertices = vertices_norm
        return mesh, center.astype(np.float32), np.float32(scale)

    def __getitem__(self, index):

        result = {}

        filename = self.data_list[index]
        uid = filename.split("/")[-1].split(".")[0]


        mesh = self.load_mesh(osp.join(self.data_path, f"{filename}"))

        mesh, norm_center, norm_scale = self.normalize_mesh(mesh)

        mesh = self.preprocess_mesh(mesh)

        mesh.faces = quad_to_triangle_mesh(mesh.faces)


        mesh = refine_large_faces(mesh, self.refine_ratio)

        result['vertices'] = mesh.vertices
        result['faces'] = mesh.faces


        pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts)
        result['pc'] = torch.from_numpy(pc).float()
        result["uid"] = uid
        result["vertices"] = torch.tensor(result['vertices'], dtype=torch.float32)
        result["faces"] = torch.tensor(result['faces'], dtype=torch.int32) 
        result["norm_center"] = torch.from_numpy(norm_center)
        result["norm_scale"] = torch.tensor(norm_scale, dtype=torch.float32)
        

        patches, face_to_patch, PP_global = build_global_patches_numba(
            mesh.vertices, mesh.faces, K=self.patch_K, seed=self.patch_seed
        )
        result["patches"] = patches
        result["face_to_patch"] = torch.from_numpy(face_to_patch)

        # Store sparse matrix components for efficient transfer
        PP_global_coo = PP_global.tocoo()
        result["PP_global_row"] = torch.from_numpy(PP_global_coo.row.astype(np.int32))
        result["PP_global_col"] = torch.from_numpy(PP_global_coo.col.astype(np.int32))
        result["PP_global_data"] = torch.from_numpy(PP_global_coo.data)
        result["PP_global_shape"] = PP_global_coo.shape
        return result
