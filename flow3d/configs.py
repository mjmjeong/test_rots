from dataclasses import dataclass


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
    rot_type: str = "6d"
    use_dual_quaternion: bool = False
    init_rot_option: str = "align_quat"
    basis_type: str = "default"
    var_activation: str = "exp" # for bayesian
    rots_var_init_value: int = -1
    transls_var_init_value: int = -1
    #### initialization / timeseries
    init_base_knn_criteria: str='velocity' # timeseries or not
    num_iters_initial_optim: int = 1000 
    #### bingham
    init_opt_with_bing: bool = False


@dataclass
class GPConfig:
    epochs: int = 5000
    batch_size: int = 100000
    transls_model: str = 'MultitaskGPModel'
    rots_model: str = 'MultitaskGPModel'
    grid_size: int = 50 
    kernel: str = 'multitask'
    transls_gp_lr: float = 0.001
    rots_gp_lr: float = 0.001
    confidence_thred: float = None
    #num_tasks: 10, # basis num
    #num_inducing=300,
    #inducing_share=True,
    #transls_lengthscale=0.1,
    #rots_lengthscale=0.1,
    #transls_kernel_type=1,
    #rots_kernel_type=1,
