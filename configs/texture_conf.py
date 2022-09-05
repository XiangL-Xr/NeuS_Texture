# !/usr/bin/python3
# coding: utf-8

class Texture_Hyparams():
    def __init__(self, base_mesh, ref_mesh, out_dir):
        super(Texture_Hyparams, self).__init__()

        self.base_mesh           = base_mesh
        self.ref_mesh            = ref_mesh
        self.out_dir             = out_dir

        self.batch               = 4
        self.spp                 = 2
        self.layers              = 1
        self.iter                = 5000
        self.display_res         = None
        self.display_interval    = 0
        self.save_interval       = 200
        self.min_roughness       = 0.08
        self.custom_mip          = False

        self.random_textures     = True
        self.learn_light         = True
        self.validate            = True
        self.local_rank          = 0
        self.train_res           = [1024, 1024]
        self.texture_res         = [1024, 1024]
        
        self.mtl_override        = None                     # Override material of model
        self.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
        self.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
        self.env_scale           = 1.0                      # Env map intensity multiplier
        self.envmap              = None                     # HDR environment probe
        self.camera_space_light  = True                     # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
        self.lock_light          = False                    # Disable light optimization in the second pass
        self.lock_pos            = False                    # Disable vertex position optimization in the second pass
        self.sdf_regularizer     = 0.5                      # Weight for sdf regularizer (see paper for details)
        self.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
        self.laplace_scale       = 3000.0                   # Weight for sdf regularizer. Default is relative with large weight
        self.loss                = 'logl1'                  # choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
        self.pre_load            = True                     # Pre-load entire dataset into memory for faster training
        self.kd_min              = [0.03, 0.03, 0.03]       # Limits for kd
        self.kd_max              = [0.8, 0.8, 0.8]
        self.ks_min              = [0, 0.08, 0]             # Limits for ks
        self.ks_max              = [0.0, 1.0, 1.0]
        self.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
        self.nrm_max             = [ 1.0,  1.0,  1.0]
        self.cam_near_far        = [0.1, 1000.0]
        self.learning_rate       = [0.03, 0.001]
        self.background          = "white"
        self.display             = [{"latlong": True}, {"bsdf": "kd"}, {"bsdf": "ks"}, {"bsdf": "normal"}]
