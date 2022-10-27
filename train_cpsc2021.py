import torch
from copy import deepcopy
from torch_ecg.cfg import set_seed, DEFAULTS

from cpsc2021.dataset import CPSC2021
from cpsc2021.trainer import CPSC2021Trainer, _set_task, _MODEL_MAP
from cpsc2021.cfg import TrainCfg, ModelCfg
from .utils import count_params

def train():

    set_seed(DEFAULTS.SEED)
    CPSC2021Trainer.__DEBUG__ = False
    CPSC2021.__DEBUG__ = False

    task = "main"
    TrainCfg.db_dir = "/home/basso/data/CPSC_2021"
    ds_train = CPSC2021(TrainCfg, training=True, task=task, lazy=False)
    ds_val = CPSC2021(TrainCfg, training=False, task=task, lazy=False)
    ds_test = CPSC2021(TrainCfg, training=False, task=task, lazy=False, test=True)

    train_config = deepcopy(TrainCfg)
    train_config[task].model_name = "cnn_phc"  # "cnn", "cnn_phc"
    train_config[task].cnn_name = "resnetNS"  # "multi_scopic", "resnetNS", "densenet_vanilla"
    train_config[task].attn_name = "se"  # "se", "none"
    train_config.n_leads = 2

    _set_task(task, train_config)
    model_config = deepcopy(ModelCfg[task])
    model_config[task].cnn_name = train_config.cnn_name
    model_config[task].attn_name = train_config.attn_name
    model_config[task].seq_lab.cnn.name = model_config[task].cnn_name
    model_config[task].seq_lab.attn.name = model_config[task].attn_name
    model_config[task].seq_lab_phc.cnn.name = model_config[task].cnn_name
    model_config[task].seq_lab_phc.attn.name = model_config[task].attn_name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model_cls = _MODEL_MAP[train_config[task].model_name]
    model = model_cls(config=model_config)
    model.to(device=device)
    print(model)
    num_params = count_params(model)

    trainer = CPSC2021Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=True,
    )
    trainer.log_manager.log_message(num_params)
    trainer._setup_dataloaders(ds_train, ds_val, ds_test)
    trainer.train()
    print(f"Best metric obtained at epoch {trainer.best_epoch}: {trainer.best_eval_res}")

    eval_res = trainer.evaluate(trainer.test_loader)
    print(f"Results on test set: {eval_res}")
    trainer.log_manager.log_metrics(
        metrics=eval_res,
        step=trainer.global_step,
        epoch=trainer.best_epoch,
        part="test",
    )

    print("Done!")


if __name__ == "__main__":
    print("Start experiments for CPSC 2021")
    train()
