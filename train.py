# Library imports
import os
from copy import deepcopy
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import SyncBatchNorm
from tqdm import tqdm
import timm.optim.optim_factory as optim_factory
import wandb

# Local imports
import datasets
from config import cfg
from misc import tools
from misc.utils import *
from misc.tools import is_main_process
from model.VIC import VideoIndividualCounter
from model.points_from_den import *


class VICTrainingPipeline:
    def __init__(self, dataset_config, working_directory):
        """
        Initialize the training pipeline.

        Args:
            dataset_config: Dataset-specific configuration
            working_directory: Current working directory path
        """
        # Initialize WandB on main process only
        self._initialize_wandb()

        # Store configuration
        self.experiment_name = cfg.EXP_NAME
        self.experiment_path = cfg.EXP_PATH
        self.working_dir = working_directory

        # Initialize model
        self.network = VideoIndividualCounter(cfg, dataset_config)
        self.network_base = self.network
        self.network.cuda()

        # Store validation settings
        self.validation_frame_gap = dataset_config.VAL_FRAME_INTERVALS

        # Setup distributed training if enabled
        if cfg.distributed:
            self._setup_distributed_training()

        # Load datasets
        self._load_datasets()

        # Setup optimizer
        self._configure_optimizer()

        # Initialize training state
        self._initialize_training_state()

        # Load checkpoint if resuming
        if cfg.RESUME:
            self._load_checkpoint()

        # Load pretrained weights if specified
        if cfg.PRE_TRAIN_COUNTER:
            self._load_pretrained_weights()

    def _initialize_wandb(self):
        """Initialize Weights & Biases logging."""
        if not is_main_process():
            return

        try:
            config_dict = cfg.__dict__ if hasattr(cfg, "__dict__") else cfg.config_dict
            wandb.init(
                project="deneme",
                entity="orhankahraman0",
                config=config_dict,
                resume=cfg.RESUME,
            )
        except Exception as error:
            print(f"Warning: Failed to initialize WandB: {error}")
            print("Continuing without WandB logging.")

    def _setup_distributed_training(self):
        """Configure model for distributed training."""
        synchronized_model = SyncBatchNorm.convert_sync_batchnorm(self.network)
        self.network = torch.nn.parallel.DistributedDataParallel(
            synchronized_model, device_ids=[cfg.gpu], find_unused_parameters=False
        )
        self.network_base = self.network.module

    def _load_datasets(self):
        """Load training and validation datasets."""
        (
            self.data_loader_train,
            self.train_sampler,
            self.data_loader_val,
            self.transform_restore,
        ) = datasets.loading_data(
            cfg.DATASET, self.validation_frame_gap, cfg.distributed, is_main_process()
        )

    def _configure_optimizer(self):
        """Setup optimizer with weight decay parameter groups."""
        parameter_groups = optim_factory.param_groups_weight_decay(
            self.network_base, cfg.WEIGHT_DECAY
        )
        self.optimizer = optim.Adam(parameter_groups, lr=cfg.LR_Base)

    def _initialize_training_state(self):
        """Initialize training counters and metrics."""
        self.iteration_counter = 0
        self.current_epoch = 1
        self.total_iterations = cfg.MAX_EPOCH * len(self.data_loader_train)

        # Initialize timers
        self.timers = {"iter time": Timer(), "train time": Timer(), "val time": Timer()}

        # Initialize best model tracking
        self.best_metrics = {
            "best_model_name": "",
            "mae": 1e20,
            "mse": 1e20,
            "seq_MAE": 1e20,
            "WRAE": 1e20,
            "MIAE": 1e20,
            "MOAE": 1e20,
            "share_mae": 1e20,
            "share_mse": 1e20,
        }

    def _load_checkpoint(self):
        """Resume training from checkpoint."""
        checkpoint = torch.load(cfg.RESUME_PATH)
        self.network.load_state_dict(checkpoint["net"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["epoch"]
        self.iteration_counter = checkpoint["i_tb"]
        self.best_metrics = checkpoint["train_record"]
        self.experiment_path = checkpoint["exp_path"]
        self.experiment_name = checkpoint["exp_name"]
        print("Finish loading resume mode")

    def _load_pretrained_weights(self):
        """Load pretrained counter weights."""
        pretrained_weights = torch.load(cfg.PRE_TRAIN_COUNTER)
        current_state = self.network.state_dict()
        updated_weights = {}

        for param_name, param_value in pretrained_weights.items():
            if "Extractor" in param_name or "global_decoder" in param_name:
                has_module_prefix = "module" in param_name

                if has_module_prefix:
                    key = param_name if cfg.distributed else param_name[7:]
                else:
                    key = f"module.{param_name}" if cfg.distributed else param_name

                updated_weights[key] = param_value

        current_state.update(updated_weights)
        self.network.load_state_dict(current_state, strict=True)

    def run_training(self):
        """Execute the complete training loop."""
        for epoch_num in range(self.current_epoch, cfg.MAX_EPOCH + 1):
            self.current_epoch = epoch_num

            # Set epoch for distributed sampler
            if cfg.distributed:
                self.train_sampler.set_epoch(epoch_num)

            # Training phase
            self.timers["train time"].tic()
            self.execute_training_epoch()
            self.timers["train time"].toc(average=False)
            print(f"train time: {self.timers['train time'].diff:.2f}s")
            print("=" * 20)

            # Validation phase
            should_validate = (
                epoch_num % cfg.VAL_INTERVAL == 0 and epoch_num >= cfg.START_VAL
            )

            if should_validate and is_main_process():
                self.timers["val time"].tic()
                self.execute_validation()
                self.timers["val time"].toc(average=False)
                print(f"val time: {self.timers['val time'].diff:.2f}s")

            # Synchronize processes
            if cfg.distributed:
                torch.distributed.barrier()

    def execute_training_epoch(self):
        """Run one epoch of training."""
        self.network.train()

        # Adjust learning rate
        current_lr = adjust_learning_rate(
            self.optimizer, cfg.LR_Base, self.total_iterations, self.iteration_counter
        )

        # Track batch losses
        loss_trackers = {}

        for batch_idx, batch_data in enumerate(self.data_loader_train):
            self.timers["iter time"].tic()
            self.iteration_counter += 1

            # Unpack batch
            images, targets = batch_data

            # Move targets to GPU
            for target_idx in range(len(targets)):
                for key, value in targets[target_idx].items():
                    if torch.is_tensor(value):
                        targets[target_idx][key] = value.cuda()

            images = images.cuda()

            # Forward pass
            (
                predicted_global_density,
                ground_truth_global_density,
                predicted_shared_density,
                ground_truth_shared_density,
                predicted_inout_density,
                ground_truth_inout_density,
                loss_components,
            ) = self.network(images, targets)

            # Calculate counts
            pred_count = predicted_global_density.sum()
            gt_count = ground_truth_global_density.sum()

            # Compute total loss
            total_loss = sum(loss_components.values())

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Reduce losses across processes
            reduced_losses = reduce_dict(loss_components)

            # Update loss trackers
            for loss_name, loss_value in reduced_losses.items():
                if loss_name not in loss_trackers:
                    loss_trackers[loss_name] = AverageMeter()
                loss_trackers[loss_name].update(loss_value.item())

            # Periodic logging
            if self.iteration_counter % cfg.PRINT_FREQ == 0:
                self._log_training_progress(
                    loss_trackers,
                    current_lr,
                    pred_count,
                    gt_count,
                    predicted_global_density,
                    ground_truth_global_density,
                    reduced_losses,
                )

            # Periodic visualization
            if self.iteration_counter % 100 == 0:
                self._save_training_visualization(
                    images,
                    ground_truth_global_density,
                    predicted_global_density,
                    ground_truth_shared_density,
                    predicted_shared_density,
                    ground_truth_inout_density,
                    predicted_inout_density,
                )

    def _log_training_progress(
        self,
        loss_trackers,
        learning_rate,
        pred_count,
        gt_count,
        pred_density,
        gt_density,
        reduced_losses,
    ):
        """Log training metrics to console and WandB."""
        if not is_main_process():
            return

        self.timers["iter time"].toc(average=False)

        # Format loss string
        loss_summary = "".join(
            [
                f"[loss_{name} {tracker.avg:.4f}]"
                for name, tracker in loss_trackers.items()
            ]
        )

        # Print to console
        print(
            f"[ep {self.current_epoch}][it {self.iteration_counter}]"
            f"{loss_summary}[{self.timers['iter time'].diff:.2f}s]"
        )

        print(
            f"[cnt: gt: {gt_count.item():.1f} pred: {pred_count.item():.1f} "
            f"max_pre: {pred_density.max().item() * cfg_data.DEN_FACTOR:.1f} "
            f"max_gt: {gt_density.max().item() * cfg_data.DEN_FACTOR:.1f}]  "
        )

        # Log to WandB
        try:
            wandb_metrics = {
                "train/lr": learning_rate,
                "train/iter_time": self.timers["iter time"].diff,
                "train/gt_global_cnt": gt_count.item(),
                "train/pre_global_cnt": pred_count.item(),
                "train/max_pre": pred_density.max().item() * cfg_data.DEN_FACTOR,
                "train/max_gt": gt_density.max().item() * cfg_data.DEN_FACTOR,
                "epoch": self.current_epoch,
                "iter": self.iteration_counter,
            }

            # Add individual losses
            for loss_name, loss_value in reduced_losses.items():
                wandb_metrics[f"train/loss_{loss_name}"] = loss_value.item()

            wandb.log(wandb_metrics, step=self.iteration_counter)
        except Exception as error:
            print(f"WandB logging error: {error}")

    def _save_training_visualization(
        self,
        images,
        gt_global,
        pred_global,
        gt_shared,
        pred_shared,
        gt_inout,
        pred_inout,
    ):
        """Save visualization of training predictions."""
        process_rank = int(os.environ.get("RANK", 0))
        output_path = os.path.join(
            self.experiment_path, self.experiment_name, "training_visual"
        )

        save_visual_results(
            [
                images,
                gt_global,
                pred_global,
                gt_shared,
                pred_shared,
                gt_inout,
                pred_inout,
            ],
            self.transform_restore,
            output_path,
            self.iteration_counter,
            process_rank,
        )

    def execute_validation(self):
        """Run validation on all scenes."""
        self.network.eval()

        # Initialize metrics
        global_errors = {"mae": AverageMeter(), "mse": AverageMeter()}

        predictions_by_scene = []
        ground_truth_by_scene = []

        # Process each scene
        for scene_idx, (scene_id, scene_dataset) in enumerate(self.data_loader_val):
            scene_predictions, scene_ground_truth = self._validate_scene(
                scene_idx, scene_id, scene_dataset, global_errors
            )

            predictions_by_scene.append(scene_predictions)
            ground_truth_by_scene.append(scene_ground_truth)

        # Compute aggregate metrics
        mae, mse, wrae, miae, moae, count_results = compute_metrics_all_scenes(
            predictions_by_scene, ground_truth_by_scene, 1
        )

        # Display results
        print(
            f"MAE: {mae.data:.2f}, MSE: {mse.data:.2f}  "
            f"WRAE: {wrae.data:.2f} WIAE: {miae.data:.2f} WOAE: {moae.data:.2f}"
        )

        # Log to WandB
        self._log_validation_metrics(mae, mse, wrae, miae, moae)

        print("Pre vs GT:", count_results)

        # Update best model
        frame_mae = global_errors["mae"].avg
        frame_mse = np.sqrt(global_errors["mse"].avg)

        self.best_metrics = update_model(
            self,
            {
                "mae": frame_mae,
                "mse": frame_mse,
                "seq_MAE": mae,
                "WRAE": wrae,
                "MIAE": miae,
                "MOAE": moae,
            },
        )

        print_NWPU_summary_det(
            self,
            {
                "mae": frame_mae,
                "mse": frame_mse,
                "seq_MAE": mae,
                "WRAE": wrae,
                "MIAE": miae,
                "MOAE": moae,
            },
        )

        torch.cuda.empty_cache()

    def _validate_scene(self, scene_idx, scene_name, scene_data, error_metrics):
        """Validate a single scene."""
        progress_bar = tqdm(scene_data)
        video_length = len(scene_data) + self.validation_frame_gap

        # Initialize prediction and ground truth dictionaries
        scene_predictions = {
            "id": scene_idx,
            "time": video_length,
            "first_frame": 0,
            "inflow": [],
            "outflow": [],
        }

        scene_ground_truth = {
            "id": scene_idx,
            "time": video_length,
            "first_frame": 0,
            "inflow": [],
            "outflow": [],
        }

        visualization_maps = []
        image_list = []

        # Process each frame
        for frame_idx, frame_data in enumerate(progress_bar):
            # Determine if this is a keyframe
            is_last_frame = frame_idx == len(scene_data) - 1
            is_keyframe = (frame_idx % self.validation_frame_gap == 0) or is_last_frame

            if not is_keyframe:
                continue

            # Process keyframe
            image, label = frame_data

            # Move labels to GPU
            for label_idx in range(len(label)):
                for key, value in label[label_idx].items():
                    if torch.is_tensor(value):
                        label[label_idx][key] = value.cuda()

            image = image.cuda()

            # Inference
            with torch.no_grad():
                # Pad image to multiple of 32
                batch_size, channels, height, width = image.shape
                pad_height = (32 - height % 32) if height % 32 != 0 else 0
                pad_width = (32 - width % 32) if width % 32 != 0 else 0

                if pad_height > 0 or pad_width > 0:
                    padding = (0, pad_width, 0, pad_height)
                    image = F.pad(image, padding, "constant")

                height, width = image.size(2), image.size(3)
                placeholder = torch.zeros((1, height, width)).cuda()

                # Forward pass
                model_to_use = self.network.module if cfg.distributed else self.network
                (
                    pred_global_den,
                    gt_global_den,
                    pred_shared_den,
                    gt_shared_den,
                    pred_inout_den,
                    gt_inout_den,
                    _,
                ) = model_to_use(image, label)

                # Clip negative predictions
                pred_inout_den[pred_inout_den < 0] = 0

                # Calculate counts
                gt_count = gt_global_den[0].sum().item()
                pred_count = pred_global_den[0].sum().item()

                # Update error metrics
                frame_mae = abs(gt_count - pred_count)
                frame_mse = (gt_count - pred_count) ** 2
                error_metrics["mae"].update(frame_mae)
                error_metrics["mse"].update(frame_mse)

                # Store first frame count
                if frame_idx == 0:
                    scene_predictions["first_frame"] = pred_count
                    scene_ground_truth["first_frame"] = gt_count

                # Store inflow/outflow
                scene_predictions["inflow"].append(pred_inout_den[1].sum().item())
                scene_predictions["outflow"].append(pred_inout_den[0].sum().item())
                scene_ground_truth["inflow"].append(gt_inout_den[1].sum().item())
                scene_ground_truth["outflow"].append(gt_inout_den[0].sum().item())

                # Prepare visualization
                if frame_idx % self.validation_frame_gap == 0:
                    current_image = image[0]
                    current_gt_global = gt_global_den[0]
                    current_pred_global = pred_global_den[0]

                    # Handle temporal dependencies
                    if frame_idx == 0:
                        prev_gt_shared = deepcopy(placeholder)
                        prev_pred_shared = deepcopy(placeholder)
                        prev_gt_in = deepcopy(placeholder)
                        prev_pred_in = deepcopy(placeholder)
                    else:
                        prev_gt_shared = previous_gt_shared[1]
                        prev_pred_shared = previous_pred_shared[1]
                        prev_gt_in = previous_gt_inout[1]
                        prev_pred_in = previous_pred_inout[1]

                    next_gt_shared = gt_shared_den[0]
                    next_pred_shared = pred_shared_den[0]
                    current_gt_out = gt_inout_den[0]
                    current_pred_out = pred_inout_den[0]

                    # Stack visualization maps
                    vis_map = torch.stack(
                        [
                            current_gt_global,
                            current_pred_global,
                            prev_gt_shared,
                            prev_pred_shared,
                            prev_gt_in,
                            prev_pred_in,
                            next_gt_shared,
                            next_pred_shared,
                            current_gt_out,
                            current_pred_out,
                        ],
                        dim=0,
                    )

                    visualization_maps.append(vis_map)
                    image_list.append(current_image)

                    # Store for next iteration
                    previous_gt_shared = gt_shared_den
                    previous_pred_shared = pred_shared_den
                    previous_gt_inout = gt_inout_den
                    previous_pred_inout = pred_inout_den

                    # Handle last frame pair
                    is_final_pair = (frame_idx + self.validation_frame_gap) > (
                        len(scene_data) - 1
                    )
                    if is_final_pair:
                        final_vis_map = torch.stack(
                            [
                                gt_global_den[1],
                                pred_global_den[1],
                                gt_shared_den[1],
                                pred_shared_den[1],
                                gt_inout_den[1],
                                pred_inout_den[1],
                                deepcopy(placeholder),
                                deepcopy(placeholder),
                                deepcopy(placeholder),
                                deepcopy(placeholder),
                            ],
                            dim=0,
                        )

                        visualization_maps.append(final_vis_map)
                        image_list.append(image[1])

        # Save visualizations
        visualization_stack = torch.stack(visualization_maps, dim=0)
        process_rank = int(os.environ.get("RANK", 0)) if cfg.distributed else 0

        save_test_visual(
            visualization_stack,
            image_list,
            scene_name,
            self.transform_restore,
            os.path.join(
                self.experiment_path, self.experiment_name, "val_visual", scene_name
            ),
            self.current_epoch,
            process_rank,
        )

        # Log validation images to WandB
        self._log_validation_images(scene_name, process_rank)

        return scene_predictions, scene_ground_truth

    def _log_validation_metrics(self, mae, mse, wrae, miae, moae):
        """Log validation metrics to WandB."""
        try:
            metrics = {
                "val/MAE": mae.item() if torch.is_tensor(mae) else mae,
                "val/MSE": mse.item() if torch.is_tensor(mse) else mse,
                "val/WRAE": wrae.item() if torch.is_tensor(wrae) else wrae,
                "val/MIAE": miae.item() if torch.is_tensor(miae) else miae,
                "val/MOAE": moae.item() if torch.is_tensor(moae) else moae,
                "epoch": self.current_epoch,
            }
            wandb.log(metrics, step=self.iteration_counter)
        except Exception as error:
            print(f"WandB val logging error: {error}")

    def _log_validation_images(self, scene_name, process_rank):
        """Log validation visualization images to WandB."""
        try:
            image_filename = f"{process_rank}_{self.current_epoch}_visual.jpg"
            image_path = os.path.join(
                self.experiment_path,
                self.experiment_name,
                "val_visual",
                scene_name,
                image_filename,
            )

            if os.path.exists(image_path):
                wandb.log(
                    {f"val/visual_{scene_name}": wandb.Image(image_path)},
                    step=self.iteration_counter,
                )
        except Exception as error:
            print(f"WandB val image log error: {error}")


def main():
    """Main entry point for training."""
    # Setup environment
    tools.init_distributed_mode(cfg)
    tools.set_randomseed(cfg.SEED + tools.get_rank())

    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Load dataset configuration
    dataset_module = import_module(f"datasets.setting.{cfg.DATASET}")
    dataset_config = dataset_module.cfg_data

    # Make cfg_data globally accessible for logging
    global cfg_data
    cfg_data = dataset_config

    # Initialize and run training
    current_dir = os.path.dirname(os.path.realpath(__file__))
    training_pipeline = VICTrainingPipeline(dataset_config, current_dir)
    training_pipeline.run_training()


if __name__ == "__main__":
    main()
