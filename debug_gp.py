import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Annotated
from typing import Literal

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
from gp_module import Motion_GP

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
    work_dir: str
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
    )
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    motion: MotionConfig
    gp: GPConfig
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
    project: str = "som-debug"
    tags: list[str] = field(default_factory=list)
    build_init: str | None = None


def main(cfg: TrainConfig):
    # import data    
    root_dir = 'observation/tmp_asset/'
    type_ = 'opt'
    debugging_set = 1000

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

    motion_gp = Motion_GP(cfg)
    canonical_idx = (confidence>0.5).sum(0).argmax().item()
    train_x, train_y = motion_gp.get_dataset(transls, rots, canonical_idx=canonical_idx, confidence=confidence)
    # run
    os.makedirs(cfg.exp_name, exist_ok=True)
    
    others={}
    data_scaled = motion_gp.bbox_normalize(transls, prefix='transls')
    if motion_gp.delta_cano:
        data_scaled = data - motion_gp.transls_cano

    others['conf'] = confidence
    others['traj'] = data_scaled

    # Print the jitter setting for float64
#        print(f"Cholesky jitter (float64): {jitter_setting.value(torch.float64)}")
    # Print the jitter setting for float32
#        print(f"Cholesky jitter (float32): {jitter_setting.value(torch.float32)}")

    motion_gp.init_gp(train_x, train_y, others=others)
    motion_gp.fitting_gp(train_x, train_y, skip_rots=True)
    motion_gp.save_csv("search_results.csv")

if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
