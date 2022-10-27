"""
"""

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import textwrap
from collections import OrderedDict
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import str2bool, get_date_str
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from .aux_metrics import compute_main_task_metric
from .cfg import BaseCfg
from .dataset import CPSC2021
from .model import ECG_SEQ_LAB_NET_CPSC2021
from .phc_model import ECG_SEQ_LAB_NET_CPSC2021_PHC

if BaseCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2021Trainer",
]


class CPSC2021Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CPSC2021Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        model: Module,
            the model to be trained
        model_config: dict,
            the configuration of the model,
            used to keep a record in the checkpoints
        train_config: dict,
            the configuration of the training,
            including configurations for the data loader, for the optimization, etc.
            will also be recorded in the checkpoints.
            `train_config` should at least contain the following keys:
                "monitor": str,
                "loss": str,
                "n_epochs": int,
                "batch_size": int,
                "learning_rate": float,
                "lr_scheduler": str,
                    "lr_step_size": int, optional, depending on the scheduler
                    "lr_gamma": float, optional, depending on the scheduler
                    "max_lr": float, optional, depending on the scheduler
                "optimizer": str,
                    "decay": float, optional, depending on the optimizer
                    "momentum": float, optional, depending on the optimizer
        device: torch.device, optional,
            the device to be used for training
        lazy: bool, default True,
            whether to initialize the data loader lazily
        """
        super().__init__(model, CPSC2021, model_config, train_config, device, lazy)

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        """

        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset
        test_dataset: Dataset, optional,
            the test dataset
        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=True,
                lazy=False,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=False,
                lazy=False,
            )

        if test_dataset is None:
            test_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=False,
                lazy=False,
            )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        num_workers = 4

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )


    def train(self) -> OrderedDict:
        """ """
        self._setup_optimizer()

        self._setup_scheduler()

        self._setup_criterion()

        msg = textwrap.dedent(
            f"""
            Starting training:
            ------------------
            Epochs:          {self.n_epochs}
            Batch size:      {self.batch_size}
            Learning rate:   {self.lr}
            Training size:   {self.n_train}
            Validation size: {self.n_val}
            Device:          {self.device.type}
            Optimizer:       {self.train_config.optimizer}
            Dataset classes: {self.train_config.classes}
            -----------------------------------------
            """
        )
        self.log_manager.log_message(msg)

        start_epoch = self.epoch
        for _ in range(start_epoch, self.n_epochs):
            # train one epoch
            self.model.train()
            self.epoch_loss = 0
            with tqdm(
                total=self.n_train,
                desc=f"Epoch {self.epoch}/{self.n_epochs}",
                unit="signals",
            ) as pbar:
                self.log_manager.epoch_start(self.epoch)
                # train one epoch
                self.train_one_epoch(pbar)

                # evaluate on train set, if debug is True
                if self.train_config.debug:
                    eval_train_res = self.evaluate(self.val_train_loader)
                    print(eval_train_res)
                    self.log_manager.log_metrics(
                        metrics=eval_train_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="train",
                    )
                # evaluate on val set
                eval_res = self.evaluate(self.val_loader)
                self.log_manager.log_metrics(
                    metrics=eval_res,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="val",
                )

                # update best model and best metric if monitor is set
                if self.train_config.monitor is not None:
                    if eval_res[self.train_config.monitor] > self.best_metric:
                        self.best_metric = eval_res[self.train_config.monitor]
                        self.best_state_dict = self._model.state_dict()
                        self.best_eval_res = deepcopy(eval_res)
                        self.best_epoch = self.epoch
                        self.pseudo_best_epoch = self.epoch
                    elif self.train_config.early_stopping:
                        if (
                            eval_res[self.train_config.monitor]
                            >= self.best_metric
                            - self.train_config.early_stopping.min_delta
                        ):
                            self.pseudo_best_epoch = self.epoch
                        elif (
                            self.epoch - self.pseudo_best_epoch
                            >= self.train_config.early_stopping.patience
                        ):
                            msg = f"early stopping is triggered at epoch {self.epoch}"
                            self.log_manager.log_message(msg)
                            break

                    msg = textwrap.dedent(
                        f"""
                        best metric = {self.best_metric},
                        obtained at epoch {self.best_epoch}
                    """
                    )
                    self.log_manager.log_message(msg)

                    # save checkpoint
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}_metric_{eval_res[self.train_config.monitor]:.2f}"
                else:
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}"
                save_filename = f"{self.save_prefix}{self.epoch}_{get_date_str()}_{save_suffix}.pth.tar"
                save_path = self.train_config.checkpoints / save_filename
                if self.train_config.keep_checkpoint_max != 0:
                    self.save_checkpoint(str(save_path))
                    self.saved_models.append(save_path)
                # remove outdated models
                if len(self.saved_models) > self.train_config.keep_checkpoint_max > 0:
                    model_to_remove = self.saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except Exception:
                        self.log_manager.log_message(
                            f"failed to remove {str(model_to_remove)}"
                        )

                # update learning rate using lr_scheduler
                if self.train_config.lr_scheduler.lower() == "plateau":
                    self._update_lr(eval_res)

                self.log_manager.epoch_end(self.epoch)

            self.epoch += 1

        # save the best model
        if self.best_metric > -np.inf:
            if self.train_config.final_model_name:
                save_filename = self.train_config.final_model_name
            else:
                save_suffix = (
                    f"metric_{self.best_eval_res[self.train_config.monitor]:.2f}"
                )
                save_filename = f"BestModel_{self.save_prefix}{self.best_epoch}_{get_date_str()}_{save_suffix}.pth.tar"
            save_path = self.train_config.model_dir / save_filename
            self.save_checkpoint(path=str(save_path))
            self.log_manager.log_message(f"best model is saved at {save_path}")
        elif self.train_config.monitor is None:
            self.log_manager.log_message(
                "no monitor is set, no model is selected and saved as the best model"
            )
            self.best_state_dict = self._model.state_dict()
        else:
            raise ValueError("No best model found!")

        self.log_manager.close()

        if not self.best_state_dict:
            # in case no best model is found,
            # e.g. monitor is not set, or keep_checkpoint_max is 0
            self.best_state_dict = self._model.state_dict()

        return self.best_state_dict

    def run_one_step(
        self, *data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        data: tuple of Tensors,
            the data to be processed for training one step (batch),
            should be of the following order:
            signals, labels, *extra_tensors

        Returns
        -------
        preds: Tensor,
            the predictions of the model for the given data
        labels: Tensor,
            the labels of the given data
        """
        signals, labels, weight_masks = data
        weight_masks = weight_masks.to(device=self.device, dtype=self.dtype)
        signals = signals.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=self.dtype)
        preds = self.model(signals)
        if self.train_config.main.loss == "AsymmetricLoss":
            return preds, labels
        else:
            return preds, labels, weight_masks

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()

        all_preds = np.array([]).reshape(
            (
                0,
                self.train_config.main.input_len
                // self.train_config.main.reduction,
            )
        )
        all_labels = np.array([]).reshape(
            (
                0,
                self.train_config.main.input_len
                // self.train_config.main.reduction,
            )
        )
        all_weight_masks = np.array([]).reshape(
            (
                0,
                self.train_config.main.input_len
                // self.train_config.main.reduction,
            )
        )
        for signals, labels, weight_masks in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy().squeeze(
                -1
            )  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
            weight_masks = weight_masks.numpy().squeeze(
                -1
            )  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
            all_labels = np.concatenate((all_labels, labels))
            all_weight_masks = np.concatenate((all_weight_masks, weight_masks))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_output = self._model.inference(signals)

            all_preds = np.concatenate(
                (all_preds, model_output.af_mask)
            )  # or model_output.prob ?

        eval_res = compute_main_task_metric(
            mask_truths=all_labels,
            mask_preds=all_preds,
            fs=self.train_config.fs,
            reduction=self.train_config.main.reduction,
            weight_masks=all_weight_masks,
        )
        # in case possible memory leakage?
        del all_preds, all_labels, all_weight_masks

        self.model.train()
        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        # return 1 if self.train_config.task in ["rr_lstm"] else 0
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return [
            "task",
        ]

    @property
    def save_prefix(self) -> str:
        return f"task-{self.train_config.task}_{self._model.__name__}_{self.model_config.cnn_name}_epoch"

    def extra_log_suffix(self) -> str:
        return f"task-{self.train_config.task}_{super().extra_log_suffix()}_{self.model_config.cnn_name}"


def get_args(**kwargs: Any):
    """NOT checked,"""
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CPSC2021",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="the batch size for training",
        dest="batch_size",
    )
    # parser.add_argument(
    #     "-c", "--cnn-name",
    #     type=str, default="multi_scopic_leadwise",
    #     help="choice of cnn feature extractor",
    #     dest="cnn_name")
    # parser.add_argument(
    #     "-r", "--rnn-name",
    #     type=str, default="none",
    #     help="choice of rnn structures",
    #     dest="rnn_name")
    # parser.add_argument(
    #     "-a", "--attn-name",
    #     type=str, default="se",
    #     help="choice of attention structures",
    #     dest="attn_name")
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=20,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="train with more debugging information",
        dest="debug",
    )

    args = vars(parser.parse_args())

    cfg.update(args)

    return CFG(cfg)


_MODEL_MAP = {
    "seq_lab": ECG_SEQ_LAB_NET_CPSC2021,
    "seq_lab_phc": ECG_SEQ_LAB_NET_CPSC2021_PHC,
}

def _set_task(task: str, config: CFG) -> None:
    """ """
    assert task in config.tasks
    config.task = task
    for item in [
        "classes",
        "monitor",
        "final_model_name",
        "loss",
    ]:
        config[item] = config[task][item]
