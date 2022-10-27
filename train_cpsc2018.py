import torch

from copy import deepcopy
from torch_ecg.cfg import set_seed, DEFAULTS

from cpsc2018.dataset import CPSC2018
from cpsc2018.trainer import CINC2020Trainer, _MODEL_MAP
from cpsc2018.cfg import TrainCfg, ModelCfg
from utils.utils import count_params


def train():
    set_seed(DEFAULTS.SEED)
    CINC2020Trainer.__DEBUG__ = False
    CPSC2018.__DEBUG__ = False

    TrainCfg.db_dir = "/home/basso/data/PhysioNetChallenge2020/Training_WFDB"
    ds_train = CPSC2018(TrainCfg, training=True, lazy=False)
    ds_val = CPSC2018(TrainCfg, training=False, lazy=False)
    ds_test = CPSC2018(TrainCfg, training=False, lazy=False, test=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    train_config = deepcopy(TrainCfg)
    train_config.model_name = "cnn_phc"  # "cnn", "cnn_phc"
    train_config.cnn_name = "resnetNS"  # "multi_scopic", "resnetNS", "densenet_vanilla"
    train_config.attn_name = "se"  # "se", "none"
    train_config.n_leads = len(train_config.leads)

    model_config = deepcopy(ModelCfg)
    model_config.cnn.name = train_config.cnn_name
    model_config.attn.name = train_config.attn_name
    model_cls = _MODEL_MAP[train_config.model_name]
    model = model_cls(classes=train_config.classes, n_leads=train_config.n_leads, config=model_config)
    model.to(device=device)
    num_params = count_params(model)
    trainer = CINC2020Trainer(
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
    trainer.log_manager.log_metrics(
        metrics=eval_res,
        step=trainer.global_step,
        epoch=trainer.best_epoch,
        part="test",
    )
    print(f"Results on test set: {eval_res}")


if __name__ == "__main__":
    print("Start experiments for CPSC 2018")
    train()
