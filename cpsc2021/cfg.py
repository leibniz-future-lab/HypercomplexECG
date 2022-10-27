"""
"""

from copy import deepcopy
from pathlib import Path

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.model_configs import (  # noqa: F401
    ECG_SEQ_LAB_NET_CONFIG,
    densenet_vanilla,
    multi_scopic,
    resnetNS,
)

from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
]

print("-- Load configs for CPSC 2021")
_BASE_DIR = Path(__file__).parent.absolute()


BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(parents=True, exist_ok=True)
BaseCfg.model_dir.mkdir(parents=True, exist_ok=True)
BaseCfg.test_data_dir = _BASE_DIR / "working_dir" / "sample_data"
BaseCfg.fs = 200
BaseCfg.torch_dtype = DEFAULTS.torch_dtype
######### n for PH layers
BaseCfg.n_leads = 2  #2 or 4

BaseCfg.class_fn2abbr = {  # fullname to abbreviation
    "non atrial fibrillation": "N",
    "paroxysmal atrial fibrillation": "AFp",
    "persistent atrial fibrillation": "AFf",
}
BaseCfg.class_abbr2fn = {v: k for k, v in BaseCfg.class_fn2abbr.items()}
BaseCfg.class_fn_map = {  # fullname to number
    "non atrial fibrillation": 0,
    "paroxysmal atrial fibrillation": 2,
    "persistent atrial fibrillation": 1,
}
BaseCfg.class_abbr_map = {
    k: BaseCfg.class_fn_map[v] for k, v in BaseCfg.class_abbr2fn.items()
}

BaseCfg.bias_thr = (
    0.15 * BaseCfg.fs
)  # rhythm change annotations onsets or offset of corresponding R peaks
BaseCfg.beat_ann_bias_thr = 0.1 * BaseCfg.fs  # half width of broad qrs complex
BaseCfg.beat_winL = 250 * BaseCfg.fs // 1000  # corr. to 250 ms
BaseCfg.beat_winR = 250 * BaseCfg.fs // 1000  # corr. to 250 ms


TrainCfg = CFG()

# common confis for all training tasks
TrainCfg.fs = BaseCfg.fs
TrainCfg.n_leads = BaseCfg.n_leads
TrainCfg.data_format = "channel_first"
TrainCfg.torch_dtype = BaseCfg.torch_dtype

TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.model_dir = BaseCfg.model_dir
TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
TrainCfg.keep_checkpoint_max = 2

TrainCfg.debug = True

# least distance of an valid R peak to two ends of ECG signals
TrainCfg.rpeaks_dist2border = int(0.5 * TrainCfg.fs)  # 0.5s
TrainCfg.qrs_mask_bias = int(0.075 * TrainCfg.fs)  # bias to rpeaks

# configs of signal preprocessing
TrainCfg.normalize = CFG(
    method="z-score",
    per_channel=True,
    mean=0.0,
    std=1.0,
)
# frequency band of the filter to apply, should be chosen very carefully
TrainCfg.bandpass = CFG(
    lowcut=0.5,
    highcut=45,
    filter_type="fir",
    filter_order=int(0.3 * TrainCfg.fs),
)

# configs of data aumentation
TrainCfg.label_smooth = False
TrainCfg.random_masking = False
TrainCfg.stretch_compress = False  # stretch or compress in time axis

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 30
TrainCfg.batch_size = 64
TrainCfg.train_ratio = 0.4

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 1e-4  # 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 5

# configs of loss function
TrainCfg.loss = "AsymmetricLoss"
TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")
TrainCfg.flooding_level = 0.0  # flooding performed if positive

TrainCfg.log_step = 40

# tasks of training
TrainCfg.tasks = [
    "main",
]

# configs of model selection
for t in TrainCfg.tasks:
    TrainCfg[t] = CFG()

TrainCfg.main.final_model_name = None
TrainCfg.main.model_name = "seq_lab_phc"  # "seq_lab", "seq_lab_phc"
TrainCfg.main.reduction = 1
TrainCfg.main.cnn_name = "multi_scopic"  # "multi_scopic", "resnetNS", "densenet_vanilla"
TrainCfg.main.rnn_name = "none"  # "none", "lstm"
TrainCfg.main.attn_name = "se"  # "none", "se", "gc", "nl"
TrainCfg.main.clf_name = "mlp"  # "mlp"
TrainCfg.main.input_len = int(30 * TrainCfg.fs)
TrainCfg.main.overlap_len = int(15 * TrainCfg.fs)
TrainCfg.main.critical_overlap_len = int(25 * TrainCfg.fs)
TrainCfg.main.classes = [
    "af",
]
TrainCfg.main.monitor = "neg_masked_bce"  # "main_score", "neg_masked_bce"  # monitor for determining the best model
TrainCfg.main.loss = "AsymmetricLoss"
TrainCfg.main.loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype
_BASE_MODEL_CONFIG.fs = BaseCfg.fs
_BASE_MODEL_CONFIG.n_leads = BaseCfg.n_leads

ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

for t in TrainCfg.tasks:
    ModelCfg[t] = deepcopy(_BASE_MODEL_CONFIG)
    ModelCfg[t].task = t

ModelCfg.main.input_len = TrainCfg.main.input_len
ModelCfg.main.classes = TrainCfg.main.classes
ModelCfg.main.model_name = TrainCfg.main.model_name
ModelCfg.main.cnn_name = TrainCfg.main.cnn_name
ModelCfg.main.rnn_name = TrainCfg.main.rnn_name
ModelCfg.main.attn_name = TrainCfg.main.attn_name
ModelCfg.main.clf_name = TrainCfg.main.clf_name

# the following is a comprehensive choices for different choices of main task
ModelCfg.main.seq_lab = deepcopy(ECG_SEQ_LAB_NET_CONFIG)
ModelCfg.main.seq_lab.fs = BaseCfg.fs
ModelCfg.main.seq_lab.reduction = TrainCfg.main.reduction
ModelCfg.main.seq_lab.cnn.name = ModelCfg.main.cnn_name
ModelCfg.main.seq_lab.rnn.name = ModelCfg.main.rnn_name
ModelCfg.main.seq_lab.attn.name = ModelCfg.main.attn_name
ModelCfg.main.seq_lab.clf.name = ModelCfg.main.clf_name

ModelCfg.main.seq_lab.cnn.multi_scopic = adjust_cnn_filter_lengths(
    ModelCfg.main.seq_lab.cnn.multi_scopic, ModelCfg.main.seq_lab.fs
)
ModelCfg.main.seq_lab.cnn.multi_scopic.filter_lengths = [
    [3, 3, 3],
    [5, 5, 3],
    [9, 7, 5],
]
ModelCfg.main.seq_lab.cnn.multi_scopic.batch_norm = True

ModelCfg.main.seq_lab.cnn.multi_scopic.batch_norm = True
ModelCfg.main.seq_lab.cnn.densenet_vanilla = deepcopy(densenet_vanilla)
ModelCfg.main.seq_lab.cnn.densenet_vanilla.num_layers = [4, 4, 4, 4]

ModelCfg.main.seq_lab_phc = deepcopy(ModelCfg.main.seq_lab)
