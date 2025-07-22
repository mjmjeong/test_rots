from dataclasses import dataclass, field



@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2
    delta_quats: float = 1e-4


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4
    rots_var: float = 1.6e-5
    transls_var: float = 1.6e-5

@dataclass
class CameraScalesLRConfig:
    camera_scales: float = 1e-4

@dataclass
class CameraPoseLRConfig:
    Rs: float = 1e-3
    ts: float = 1e-3

@dataclass
class SceneLRConfig:
    fg: FGLRConfig
    bg: BGLRConfig
    motion_bases: MotionLRConfig
    camera_poses: CameraPoseLRConfig
    camera_scales: CameraScalesLRConfig


@dataclass
class LossesConfig:
    w_rgb: float = 1.0
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.1
    w_depth_grad: float = 1
    w_track: float = 2.0
    w_mask: float = 1.0
    w_smooth_bases: float = 0.1
    w_smooth_bases_rots: float = 1.0
    w_smooth_bases_transls: float = 2.0
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0

    w_commit: float = 0.0
    w_recon: float = 0.0
    w_kl_rots: float = 0.0
    w_kl_transls: float = 0.0
    
    # w_smooth_bases: float = 0.0
    # w_smooth_tracks: float = 0.0
    # w_scale_var: float = 0.0
    # w_z_accel: float = 0.0

    w_bing_commit: float = 0.0
    w_bing_recon: float = 0.0
    w_bing_intensity: float = 0.0
    w_bing_smooth: float = 0.0


@dataclass
class OptimizerConfig:
    max_steps: int = 5000
    ## Adaptive gaussian control
    warmup_steps: int = 200
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15                                                                                                                                                                                                                                 
    ### Prior update
    cache_prior_every: int = 1

@dataclass
class MotionConfig:
    #########################################
    # preprocessing: tracks 3d
    #########################################
    use_gp_preprocessing: bool = False
    num_iters_initial_optim: int = 1000 
    filling_missing_tracks3d: str = 'interp' # interp, gp
    init_base_knn_criteria: str = "velocity" # chronos_mean, chronos_first, ...
    #########################################
    # basis type
    #########################################
    rot_type: str = "6d"
    init_rot_option: str = "align_quat"
    use_dual_quaternion: bool = False
    basis_type: str = "default"
    var_activation: str = "exp" # for bayesian
    rots_var_init_value: int = -1
    transls_var_init_value: int = -1
    #### bingham
    init_opt_with_bing: bool = False

@dataclass
class GPConfig:
    transls_model: str = 'IndependentVariationalGPModel'
    rots_model: str = 'IndependentVariationalGPModel'
    #########################################
    # input x (uncertain input)
    #########################################
    delta_cano: bool = False
    x_rsample: str = 'none'
    rsample_std: float = 0.1
    #########################################
    # inducing points
    #########################################
    inducing_num: int = 100
    inducing_point_noise_scale: float = 0.0
    inducing_min: float = -1
    inducing_max: float = 1    
    nx: int = 6
    nt: int = 200
    inducing_method: str = 'vel_chronos_kmeans' # 'RX-grid'
    inducing_task_specific: bool = False
    #########################################
    # Kernel
    #########################################
    # hexplane
    combine_type: str = 'prod'
    transls_lengthscale_xy = 0.001
    transls_lengthscale_zt = 0.001
    rots_lengthscale_xy = 0.001
    rots_lengthscale_zt = 0.001
    nu_matern_xy = 1.5
    nu_matern_zt = 1.5
    # interpolation kernel
    #use_grid_kernel: bool = False
    #use_hexplane_grid_kernel: bool = False
    #grid_min: float = -1.1
    #grid_max: float = 1.1
    #grid_size: list = field(default_factory=lambda: [40, 40, 40, 160])
    # nn strategy
    #knn: int = 32
    #use_multitask: bool = True
    #########################################
    # Optimization
    #########################################
    epochs: int = 10
    batch_size: int = 20000
    transls_gp_lr: float = 0.01
    rots_gp_lr: float = 0.01
    confidence_thred: float = 0.0