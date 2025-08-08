import subprocess
import json
import optuna
import os
import time
import uuid

def objective(trial):
    # Define hyperparameter search space
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    nerf_samples = trial.suggest_categorical("num_nerf_samples", [32, 48, 64])
    proposal_samples_1 = trial.suggest_categorical("proposal_samples_1", [64, 96, 128, 256])
    proposal_samples_2 = trial.suggest_categorical("proposal_samples_2", [32, 64, 96, 128])
    interlevel_mult = trial.suggest_float("interlevel_loss_mult", 0.5, 3.0)
    distortion_mult = trial.suggest_float("distortion_loss_mult", 0.0001, 0.01, log=True)
    appearance_dim = trial.suggest_categorical("appearance_embed_dim", [16, 32, 64])
    use_embedding = trial.suggest_categorical("use_appearance_embedding", [True, False])
    use_avg_embedding = trial.suggest_categorical("use_average_appearance_embedding", [True, False])

    # Camera optimizer and scheduler
    camera_opt_lr = trial.suggest_float("camera_opt_lr", 1e-5, 1e-2, log=True)
    camera_opt_eps = trial.suggest_float("camera_opt_eps", 1e-16, 1e-8, log=True)
    camera_opt_weight_decay = trial.suggest_float("camera_opt_weight_decay", 0.0, 1e-2)
    camera_opt_sch_lr_pre_warmup = trial.suggest_float("camera_opt_sch_lr_pre_warmup", 1e-9, 1e-6, log=True)
    camera_opt_sch_lr_final = trial.suggest_float("camera_opt_sch_lr_final", 1e-5, 1e-3, log=True)
    camera_opt_sch_warmup_steps = trial.suggest_int("camera_opt_sch_warmup_steps", 0, 100)
    camera_opt_sch_max_steps = trial.suggest_int("camera_opt_sch_max_steps", 1000, 10000)

    # Field optimizer and scheduler
    field_opt_lr = trial.suggest_float("field_opt_lr", 1e-4, 1e-1, log=True)
    field_opt_eps = trial.suggest_float("field_opt_eps", 1e-16, 1e-8, log=True)
    field_opt_weight_decay = trial.suggest_float("field_opt_weight_decay", 0.0, 1e-2)
    field_sch_lr_pre_warmup = trial.suggest_float("field_sch_lr_pre_warmup", 1e-9, 1e-6, log=True)
    field_sch_lr_final = trial.suggest_float("field_sch_lr_final", 1e-5, 1e-3, log=True)
    field_sch_warmup_steps = trial.suggest_int("field_sch_warmup_steps", 0, 1000)
    field_sch_max_steps = trial.suggest_int("field_sch_max_steps", 10000, 300000)

    '''
    ### Optimizer
    # optimizer camera opt and scheduler
    # field opt and scheduler
    camera_opt_lr = 0.001 #--optimizers.camera-opt.optimizer.lr
    camera_opt_eps = 1e-15 #--optimizers.camera-opt.optimizer.eps
    camera_opt_weight_decay = 0 #--optimizers.camera-opt.optimizer.weight-decay
    camera_opt_sch_lr_pre_warmup = 1e-08 #--optimizers.camera-opt.scheduler.lr-pre-warmup
    camera_opt_sch_lr_final = 0.0001 #--optimizers.camera-opt.scheduler.lr-final
    camera_opt_sch_warmup_steps = 0 #--optimizers.camera-opt.scheduler.warmup-steps
    camera_opt_sch_max_steps = 5000 #--optimizers.camera-opt-scheduler.max-steps

    field_opt_lr= 0.01 #--optimizers.fields.optimizer.lr
    field_opt_eps = 1e-15 #--optimizers.fields.optimizer.eps
    field_opt_weight_decay = 0 #--optimizers.fields.optimizer.weight-decay
    field_sch_lr_pre_warmup= 1e-08 #--optimizers.fields.scheduler.lr-pre-warmup
    field_sch_lr_final = 0.0001 #--optimizers.fields.scheduler.lr-final
    field_sch_warmup_steps = 0 #--optimizers.fields.scheduler.warmup-steps
    field_sch_max_steps = 200000 #--optimizers.fields.scheduler.max-steps
    ###
    '''

    # Generate unique run name and paths
    run_id = uuid.uuid4().hex[:6]
    run_name = f"trial-{trial.number}-{run_id}"
    output_dir = os.path.join("outputs", "optuna_runs", run_name)
    data_dir = "./data/nerfstudio/plane"
    os.makedirs(output_dir, exist_ok=True)

    # Build ns-train command
    cmd = [
        "ns-train", "nerfacto",
        f"--experiment-name={run_name}",
        f"--output-dir={output_dir}",
        f"--data={data_dir}",
        f"--pipeline.model.hidden-dim={hidden_dim}",
        f"--pipeline.model.num-nerf-samples-per-ray={nerf_samples}",
        f"--pipeline.model.num-proposal-samples-per-ray", str(proposal_samples_1), str(proposal_samples_2),
        f"--pipeline.model.interlevel-loss-mult={interlevel_mult}",
        f"--pipeline.model.distortion-loss-mult={distortion_mult}",
        f"--pipeline.model.appearance-embed-dim={appearance_dim}",
        f"--pipeline.model.use-appearance-embedding={use_embedding}",
        f"--pipeline.model.use-average-appearance-embedding={use_avg_embedding}",
        f"--optimizers.camera-opt.optimizer.lr={camera_opt_lr}",
        f"--optimizers.camera-opt.optimizer.eps={camera_opt_eps}",
        f"--optimizers.camera-opt.optimizer.weight-decay={camera_opt_weight_decay}",
        f"--optimizers.camera-opt.scheduler.lr-pre-warmup={camera_opt_sch_lr_pre_warmup}",
        f"--optimizers.camera-opt.scheduler.lr-final={camera_opt_sch_lr_final}",
        f"--optimizers.camera-opt.scheduler.warmup-steps={camera_opt_sch_warmup_steps}",
        f"--optimizers.camera-opt.scheduler.max-steps={camera_opt_sch_max_steps}",
        f"--optimizers.fields.optimizer.lr={field_opt_lr}",
        f"--optimizers.fields.optimizer.eps={field_opt_eps}",
        f"--optimizers.fields.optimizer.weight-decay={field_opt_weight_decay}",
        f"--optimizers.fields.scheduler.lr-pre-warmup={field_sch_lr_pre_warmup}",
        f"--optimizers.fields.scheduler.lr-final={field_sch_lr_final}",
        f"--optimizers.fields.scheduler.warmup-steps={field_sch_warmup_steps}",
        f"--optimizers.fields.scheduler.max-steps={field_sch_max_steps}",
        f"--pipeline.model.use-appearance-embedding=False",
        f"--pipeline.model.use-average-appearance-embedding=False",
        "--max-num-iterations=50000",
        "--viewer.quit-on-train-completion=True",
    ]


    # Run the subprocess and catch failures
    try:
        result = subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial.number}] Nerfstudio failed.\nSTDERR:\n{e.stderr}")
        raise optuna.TrialPruned()

    import glob

    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial.number}] ns-train failed.\nSTDERR:\n{e}")
        raise optuna.TrialPruned()

    config_glob = os.path.join(output_dir, run_name, "nerfacto", "*", "config.yml")
    config_files = glob.glob(config_glob)
    if not config_files:
        print(f"[Trial {trial.number}] config.yml not found in: {config_glob}")
        raise optuna.TrialPruned()

    config_path = config_files[0]
    eval_dir = os.path.dirname(config_path)
    eval_output_path = os.path.join(eval_dir, "output.json")

    # Evaluation
    eval_cmd = [
        "ns-eval",
        f"--load-config={config_path}",
        f"--output-path={eval_output_path}"
    ]

    try:
        subprocess.run(eval_cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial.number}] ns-eval failed.\nSTDERR:\n{e}")
        raise optuna.TrialPruned()

    #Reading metrics
    metrics_path = os.path.join(eval_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[Trial {trial.number}] metrics.json not found at {metrics_path}")
        raise optuna.TrialPruned()

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        psnr = metrics["eval"]["psnr"]
    except Exception as e:
        print(f"[Trial {trial.number}] Failed to read metrics.json: {e}")
        raise optuna.TrialPruned()

    # Save trial parameters for record
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "psnr": psnr,
            "params": trial.params
        }, f, indent=2)

    trial.report(-psnr, step=1)
    if trial.should_prune():
        raise optuna.TrialPruned()

    print(f"[Trial {trial.number}] PSNR: {psnr:.2f} | Params: {trial.params}")
    return -psnr


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name="nerfacto-opt",
        sampler=sampler
    )
    study.optimize(objective, n_trials=20)
