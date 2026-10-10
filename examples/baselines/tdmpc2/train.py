"""Train the default TD-MPC2 planner for closed-path shaping."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ["LAZY_LEGACY_OP"] = "0"

import hydra
import torch
from omegaconf import OmegaConf
from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.logger import Logger, print_run
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config_traj_shape_pomdp_default", config_path=".", version_base=None)
def train(cfg):
    if cfg.env_type == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA is unavailable.")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    logger = Logger(cfg)
    env = make_envs(cfg, cfg.num_envs)
    eval_env = make_envs(cfg, cfg.num_eval_envs, is_eval=True)
    print_run(cfg)
    agent = TDMPC2(cfg)
    if logger._wandb is not None:
        logger._wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    trainer = OnlineTrainer(cfg=cfg, env=env, eval_env=eval_env, agent=agent,
                            buffer=Buffer(cfg), logger=logger)
    OmegaConf.save(cfg, cfg.work_dir / "config.yaml", resolve=True)
    trainer.train()
    env.close()
    eval_env.close()
    print("Training completed successfully")


if __name__ == "__main__":
    train()
