import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Annotated
from typing import Literal
import wandb

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig, MotionConfig, GPConfig
from flow3d.data import (
    BaseDataset,
    DavisDataConfig,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
    init_trainable_poses,
)
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server
from flow3d.params import CameraScales
from gp_module import MotionGP

torch.set_float32_matmul_precision("high")


def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@dataclass
class TrainConfig:
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    motion: MotionConfig
    gp: GPConfig
    work_dir: str = 'search/wandb_sweep'
    num_fg: int = 40_000
    num_bg: int = 100_000
    num_motion_bases: int = 10
    num_epochs: int = 500
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50
    use_2dgs: bool = False
    exp_name: str = "debug"
    project: str = "gp-offline"
    tags: list[str] = field(default_factory=list)
    build_init: str | None = None


def main(cfg: TrainConfig):
    # Initialize WandB - this will use sweep config when run by agent
    name = f"{cfg.exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run = wandb.init(
        project=cfg.project,
        name=name,
        config=asdict(cfg)
    )
    
    # CRITICAL: Update cfg with wandb.config values for sweep parameters
    # This allows WandB to override the config values during sweeps
    if wandb.config:
        guru.info("Updating config with WandB sweep parameters")
        # Update GP config parameters from wandb.config
        gp_params = [
                    'transls_model',
                    'rots_model',
                    'delta_cano',
                    'x_rsample',
                    'rsample_std',
                    'inducing_num',
                    'inducing_point_noise_scale',
                    'inducing_min',
                    'inducing_max',
                    'nx',
                    'nt',
                    'inducing_method',
                    'inducing_task_specific',
                    'combine_type',
                    'transls_lengthscale_xy',
                    'transls_lengthscale_zt',
                    'rots_lengthscale_xy',
                    'rots_lengthscale_zt',
                    'nu_matern_xy',
                    'nu_matern_zt',
                    'epochs',
                    'batch_size',
                    'transls_gp_lr',
                    'rots_gp_lr',
                    'confidence_thred'
                ]
                        
        for param in gp_params:
            wandb_key = f'gp.{param}'
            if wandb_key in wandb.config:
                setattr(cfg.gp, param, wandb.config[wandb_key])
                guru.info(f"Updated cfg.gp.{param} = {wandb.config[wandb_key]}")
        
        # Handle the special boolean flag parameter
        if 'gp.inducing_task_specific' in wandb.config:
            # WandB will provide True/False, we set it directly to the config
            cfg.gp.inducing_task_specific = wandb.config['gp.inducing_task_specific']
            guru.info(f"Updated cfg.gp.inducing_task_specific = {cfg.gp.inducing_task_specific}")
    
        # Update other top-level parameters
        top_level_params = ['num_epochs', 'batch_size', 'exp_name', 'project']
        for param in top_level_params:
            if param in wandb.config:
                setattr(cfg, param, wandb.config[param])
                guru.info(f"Updated cfg.{param} = {wandb.config[param]}")

    os.makedirs(os.path.join('search/full_wandb_sweep'), exist_ok=True)
    cfg.exp_name = os.path.join(f'search/full_wandb_sweep/{name}')
    #cfg.exp_name = name
    guru.info(f"Final config after WandB updates:\n{asdict(cfg)}")
    # import data    
    root_dir = 'observation/tmp_asset/'
    type_ = 'opt'
    debugging_set = -1

    if type_ == 'init': 
        transls = torch.load(f"{root_dir}/init_3dtraj/xyz.pt", weights_only=False).cpu()
        rots_basis = torch.load(f"{root_dir}rots_init.pt", weights_only=False)
        coef = torch.load(f"{root_dir}coef_init.pt", weights_only=False)
        rots = torch.einsum("gb,btm->gtm", coef, rots_basis)
        confidence = 1.0*torch.load(f"{root_dir}/init_3dtraj/visible.pt").cpu().unsqueeze(-1)
  
    elif type_ == 'opt':
        transls_basis = torch.load(f"{root_dir}transls_opt.pt", weights_only=False)
        rots_basis = torch.load(f"{root_dir}rots_opt.pt", weights_only=False)
        coef = torch.load(f"{root_dir}coef_opt.pt", weights_only=False)

        transls = torch.einsum("gb,btm->gtm", coef, transls_basis)
        rots = torch.einsum("gb,btm->gtm", coef, rots_basis)
        confidence = torch.randn_like(transls[:,:,0]).unsqueeze(-1)
        confidence = torch.ones_like(confidence)
    
    elif type_ == 'opt_artificial':
        transls_basis = torch.load(f"{root_dir}transls_opt.pt", weights_only=False)
        rots_basis = torch.load(f"{root_dir}rots_opt.pt", weights_only=False)
        coef = torch.load(f"{root_dir}coef_opt.pt", weights_only=False)

        transls = torch.einsum("gb,btm->gtm", coef, transls_basis)
        rots = torch.einsum("gb,btm->gtm", coef, rots_basis)
        confidence = torch.randn_like(transls[:,:,0]).unsqueeze(-1)

        N = transls.size(0)
        T = transls.size(1)
        transls = transls[0].unsqueeze(0).repeat(N,1,1)
        rots = rots[0].unsqueeze(0).repeat(N,1,1)

        noise_transls = torch.randn_like(rots[:,0,0]).reshape(N,1,1).repeat(1,T,3) * 3
        noise_rots = torch.randn_like(rots[:,0,0]).reshape(N,1,1).repeat(1,T,6) * 3

        transls += noise_transls
        rots += noise_rots

    if debugging_set > 0:
        transls = transls[:debugging_set]
        rots = rots[:debugging_set]
        confidence = confidence[:debugging_set]
        
    cfg.init_data = type_
    cfg.data_num = debugging_set
    # init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cfg.device = device

    motion_gp = MotionGP(cfg)    
    train_x, train_y = motion_gp.get_training_dataset(transls, rots, confidence_weights=confidence)
    # run
    os.makedirs(cfg.exp_name, exist_ok=True)
    
    # Print the jitter setting for float64
#        print(f"Cholesky jitter (float64): {jitter_setting.value(torch.float64)}")
    # Print the jitter setting for float32
#        print(f"Cholesky jitter (float32): {jitter_setting.value(torch.float32)}")

    if not motion_gp.is_trained:
        # Full Traj (even with messy)
        others={}
        data_scaled = motion_gp.normalize_to_bbox(transls, prefix='transls')
        if motion_gp.delta_cano:
            data_scaled = data - motion_gp.transls_cano
        others['conf'] = confidence
        others['traj'] = data_scaled
        motion_gp.initialize_gp_models(train_x, train_y, others=others)
    
#    motion_gp.is_trained = True
    motion_gp.training_gp_models(train_x, train_y, skip_rots=True)
    motion_gp.save_csv("search_results.csv")
    dict_ = motion_gp.get_state_dict()
    torch.save(dict_, 'test_ckpt_0.pth')
    gp_state_dict = torch.load('test_ckpt_0.pth')
    motion_gp2 = MotionGP(cfg)
    motion_gp2 = motion_gp2.initialize_gp_models_from_state_dict(gp_state_dict, cfg)
    
    wandb.log({'val/loss':motion_gp.optimization_stats['loss'],
                'val/lengthscale_xy':motion_gp.optimization_stats['transls_xy_lengthscale'],
                'val/lengthscale_zt':motion_gp.optimization_stats['transls_zt_lengthscale'],
                })
    wandb.finish()

def create_sweep_config():
    """Create the WandB sweep configuration"""
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val/loss',  # Make sure this metric is logged in your training loop
            'goal': 'minimize'
        },
        'program': 'debug_gp.py',
        'parameters': {
            # Fixed values for non-GP parameters
            'batch_size': {'value': 8},
            'project': {'value': 'Flow3D_GP_Sweeps'},
            'gp.epochs': {'value': 100},
            'gp.batch_size': {'values': [5000]},
            'gp.inducing_method': {'values': ['vel_chronos_kmeans']},
            'gp.transls_gp_lr': {'values': [0.1, 0.01, 0.001, 0.005, 0.0001]},
            'gp.combine_type': {'values': ['prod', 'add']},
            'gp.transls_lengthscale_xy': {'values': [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00001]},
            'gp.transls_lengthscale_zt': {'values': [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00001]},
            'gp.nu_matern_xy': {'values': [0.5, 1.5]},
            'gp.nu_matern_zt': {'values': [0.5, 1.5]},
            'gp.nx': {'values': [6]},
            'gp.nt': {'values': [200]},
        }
    }

    return sweep_config


def create_sweep_config():
    """Create the WandB sweep configuration"""
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val/loss',  # Make sure this metric is logged in your training loop
            'goal': 'minimize'
        },
        'program': 'debug_gp.py',
        'parameters': {
            # Fixed values for non-GP parameters
            'batch_size': {'value': 8},
            'project': {'value': 'full_init_search'},
            'gp.epochs': {'value': 3},
            'gp.batch_size': {'values': [5000]},
            'gp.inducing_method': {'values': ['vel_chronos_kmeans']},
            'gp.transls_gp_lr': {'values': [0.1, 0.01, 0.001, 0.005, 0.0001]},
            'gp.transls_lengthscale_xy': {'values': [0.002, 0.001, 0.0005]},
            'gp.transls_lengthscale_zt': {'values': [0.002, 0.001, 0.0005]},
            'gp.nu_matern_xy': {'values': [0.5, 1.5]},
            'gp.nu_matern_zt': {'values': [0.5, 1.5]},
            'gp.nx': {'values': [6]},
            'gp.nt': {'values': [200]},
        }
    }
    return sweep_config

if __name__ == "__main__":
    # Check if this is being called by wandb agent

#   guru.info("Running as WandB sweep agent")
    main(tyro.cli(TrainConfig))
    """
    if 'WANDB_SWEEP_ID' in os.environ:
        # Being called by wandb agent - run with tyro.cli
        guru.info("Running as WandB sweep agent")
        main(tyro.cli(TrainConfig))

    else:
        # Generate sweep configuration
        guru.info("Generating WandB sweep configuration...")
        sweep_config = create_sweep_config()
        
        sweep_config_path = "sweep_config_gp_only.yaml"
        with open(sweep_config_path, 'w') as f:
            yaml.dump(sweep_config, f, sort_keys=False)
        
        guru.info(f"WandB sweep configuration saved to {sweep_config_path}")
        guru.info("\n--- WandB Setup & Run Instructions ---")
        guru.info("1. Log in to WandB (if not already): `wandb login`")
        guru.info("2. Initialize the sweep: `wandb sweep sweep_config_gp_only.yaml`")
        guru.info("   (This will give you a SWEEP_ID)")
        guru.info("3. Run agents: `wandb agent <SWEEP_ID>`")
        guru.info("\nTo test locally without sweep, run:")
        guru.info("python script.py --exp_name test_run")
    """