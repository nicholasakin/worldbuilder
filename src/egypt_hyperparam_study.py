import os
from datetime import datetime
from pathlib import Path

import gc
import optuna
from optuna.trial._trial import Trial
import torch

from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.scripts.eval import ComputePSNR

from nerfstudio.scripts.train import train_loop


def train_test_model(config: TrainerConfig, method_name: str, timestamp: str) -> float:
    local_rank = 0
    world_size = 1
    global_rank = 0

    metrics_dict = train_loop(
        local_rank=local_rank,
        world_size=world_size,
        config=config,
        global_rank=global_rank
    )

    config_path = Path(os.path.join("outputs", "Egypt", method_name, timestamp, "config.yml"))
    psnr = ComputePSNR(
        load_config=config_path,
        output_path=Path(os.path.join(config_path.parent, "output.json"))
    ).main()
    return psnr


def objective(trial: Trial):
    nerfacto_config = NerfactoModelConfig()

    ####################
    nerfacto_config.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    nerfacto_config.num_nerf_samples_per_ray = trial.suggest_categorical("num_nerf_samples", [32, 48, 64])
    nerfacto_config.num_proposal_samples_per_ray = (
        trial.suggest_categorical("proposal_samples_1", [64, 96, 128, 256]),
        trial.suggest_categorical("proposal_samples_2", [32, 64, 96, 128])
    )
    nerfacto_config.interlevel_loss_mult = trial.suggest_float("interlevel_loss_mult", 0.5, 3.0)
    nerfacto_config.distortion_loss_mult = trial.suggest_float("distortion_loss_mult", 0.0001, 0.01, log=True)
    nerfacto_config.appearance_embed_dim = trial.suggest_categorical("appearance_embed_dim", [16, 32, 64])
    nerfacto_config.use_appearance_embedding = trial.suggest_categorical("use_appearance_embedding", [True, False])
    nerfacto_config.use_average_appearance_embedding = trial.suggest_categorical("use_average_appearance_embedding", [True, False])
    ####################

    nerfacto_config.near_plane = 0.05
    nerfacto_config.far_plane = 1000.0
    nerfacto_config.background_color = "last_sample"
    nerfacto_config.hidden_dim_color = 64
    nerfacto_config.hidden_dim_transient = 64
    nerfacto_config.num_levels = 16
    nerfacto_config.base_res = 16
    nerfacto_config.proposal_update_every = 5
    nerfacto_config.proposal_warmup = 5000
    nerfacto_config.num_proposal_iterations = 2
    proposal_net_args_list = [
        {
            "hidden_dim": 16, 
            "log2_hashmap_size": 17, 
            "num_levels": 5, 
            "max_res": 128, 
            "use_linear": False
        },
        {
            "hidden_dim": 16, 
            "log2_hashmap_size": 17, 
            "num_levels": 5, 
            "max_res": 256, 
            "use_linear": False
        },
    ]
    nerfacto_config.proposal_net_args_list = proposal_net_args_list
    nerfacto_config.proposal_initial_sampler = "piecewise"
    nerfacto_config.use_proposal_weight_anneal = True
    nerfacto_config.interlevel_loss_mult = 1.0
    nerfacto_config.distortion_loss_mult = 0.002
    nerfacto_config.orientation_loss_mult = 0.0001
    nerfacto_config.pred_normal_loss_mult = 0.001
    nerfacto_config.use_proposal_weight_anneal = True
    nerfacto_config.use_appearance_embedding = True
    nerfacto_config.use_average_appearance_embedding = True
    nerfacto_config.proposal_weights_anneal_slope = 10.0
    nerfacto_config.proposal_weights_anneal_max_num_iters = 1000
    nerfacto_config.use_single_jitter = True
    nerfacto_config.predict_normals = False
    nerfacto_config.disable_scene_contraction = False
    nerfacto_config.use_gradient_scaling = False
    nerfacto_config.implementation = "tcnn"
    nerfacto_config.appearance_embed_dim = 32
    nerfacto_config.average_init_density = 0.01

    data_path = Path(r"C:\Users\asoli\Documents\Summer_2025\Computer_Vision\Project\nerfstudio\data\nerfstudio\Egypt")
    data_parser = NerfstudioDataParserConfig(data=data_path)
    pdmc = ParallelDataManagerConfig(
        dataparser=data_parser,
        data=data_path
    )
    pipeline_config = VanillaPipelineConfig(
        model=nerfacto_config,
        datamanager=pdmc,
    )
    method_name = f"nerfacto"
    timestamp = datetime.now().strftime("%Y-%m-%d_%S-%f")
    os.mkdir(os.path.join("outputs", "Egypt", method_name, timestamp))
    config = TrainerConfig(pipeline=pipeline_config)
    config.method_name = method_name
    config.timestamp = timestamp

    optimizers = {
        'proposal_networks': {
            'optimizer': AdamOptimizerConfig(eps=1e-15, lr=0.01),
            'scheduler': ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200_000),
        },
        'fields': {
            'optimizer': AdamOptimizerConfig(eps=1e-15, lr=0.01),
            'scheduler': ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200_000),
        },
        'camera_opt': {
            'optimizer': AdamOptimizerConfig(eps=1e-15, lr=0.001),
            'scheduler': ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=5_000),
        },
    }
    config.optimizers = optimizers
    config.machine.device_type = "cuda"
    config.machine.dist_url = "auto"
    config.machine.machine_rank = 0
    config.machine.num_devices = 1
    config.machine.num_machines = 1
    config.machine.seed = 42
    config.vis = 'none'
    config.steps_per_save = 2000
    config.max_num_iterations = 50_000
    config.mixed_precision = True
    config.data = data_path

    config.save_config()

    psnr = train_test_model(config=config, method_name=method_name, timestamp=timestamp)

    torch.cuda.empty_cache()
    gc.collect()

    return psnr

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)

if __name__ == "__main__":
    main()
