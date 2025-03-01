import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from dataclasses import dataclass
from dataclasses import asdict

from datetime import datetime
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from tqdm import tqdm



from flow3d.data import get_train_val_datasets
from flow3d.renderer import Renderer

torch.set_float32_matmul_precision("high")


@dataclass
class LoadConfig:
    work_dir: str
    port: int = 8890

def main(cfg: LoadConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir=cfg.work_dir,
        port=cfg.port,
    )

    assert os.path.exists(ckpt_path)
    save_dir = f"{cfg.work_dir}/observation"
    os.makedirs(save_dir, exist_ok=True)
    # motion-basis
    motion_num = renderer.model.num_motion_bases
    motion_bases = renderer.model._modules['motion_bases']
    rots = motion_bases._modules['params'].rots
    torch.save(rots.cpu().detach(), os.path.join(save_dir, 'rot.pts'))
    transls = motion_bases._modules['params'].transls
    torch.save(transls.cpu().detach(), os.path.join(save_dir,'transls.pts'))
    # coeff
    coeff = renderer.model._modules['fg']._modules['params'].motion_coefs
    torch.save(coeff.cpu().detach(), os.path.join(save_dir,'fg_coeff.pts'))
    opacities = renderer.model.fg.get_opacities()
    torch.save(opacities.cpu().detach(), os.path.join(save_dir,'fg_opacity.pts'))

    bg_opacities = renderer.model.bg.get_opacities()
    torch.save(bg_opacities.cpu().detach(), os.path.join(save_dir,'bg_opacity.pts'))
    

if __name__ == "__main__":
    main(tyro.cli(LoadConfig))
    
    # example: python observation/load_motion_params_save.py --work-dir outputs/spin_base