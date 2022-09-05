# !/usr/bin/python3
# coding: utf-8
# author: lixiang

import torch
import xatlas
import numpy as np

from render import mesh
from render import render
from render import regularizer

class CustomGeometry():
    def __init__(self, base_mesh, FLAGS, init_uvs=False):
        super(CustomGeometry, self).__init__()

        self.FLAGS = FLAGS
        self.init_uvs = init_uvs
        self.verts = torch.tensor(base_mesh.vertices, dtype=torch.float32, device='cuda')
        self.faces = torch.tensor(base_mesh.faces, dtype=torch.long, device='cuda')
        print("=> Base mesh has %d triangles and %d vertices." % (self.faces.shape[0], self.verts.shape[0]))

        if self.init_uvs:
            print('=> Initialize the uvs by xatlas...')
            self.uvs, self.uvs_idx = self.getUVmap()
    
    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    def getUVmap(self):
        # get uvs with xatlas
        v_pos = self.verts.detach().cpu().numpy()
        t_pos_idx = self.faces.detach().cpu().numpy()
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

        # Convert to tensors
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
        uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
        uvs_idx = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

        return uvs, uvs_idx

    def getMesh(self, material):

        if self.init_uvs:
            imesh = mesh.Mesh(self.verts, self.faces, v_tex=self.uvs, t_tex_idx=self.uvs_idx, material=material) 
        else:
            # get uvs with xatlas
            v_pos = self.verts.detach().cpu().numpy()
            t_pos_idx = self.faces.detach().cpu().numpy()
            vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

            # Convert to tensors
            indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
            uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
            uvs_idx = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

            imesh = mesh.Mesh(self.verts, self.faces, v_tex=uvs, t_tex_idx=uvs_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        
        return imesh
    
    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'],
                                  spp=target['spp'], msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)
        #print('--buffers: ', buffers)
        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.005

        return img_loss, reg_loss

class Init_Geometry():
    def __init__(self, base_mesh):
        super(Init_Geometry, self).__init__()

        self.verts = torch.tensor(base_mesh.vertices, dtype=torch.float32, device='cuda')
        self.faces = torch.tensor(base_mesh.faces, dtype=torch.long, device='cuda')
    
    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    def InitMesh(self):
        return self.verts, self.faces
        

class CustomMesh():
    def __init__(self, initial_guess, FLAGS):
        super(CustomMesh, self).__init__()

        self.FLAGS = FLAGS

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("=> Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material
        imesh = mesh.Mesh(base=self.mesh)
        
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer. 
        if self.FLAGS.laplace == "absolute":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)
        elif self.FLAGS.laplace == "relative":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)                

        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.01

        return img_loss, reg_loss
