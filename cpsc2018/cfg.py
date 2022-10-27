"""
configurations for signal preprocess, feature extraction, training, etc.
along with some constants

"Brady", "LAD", "RAD", "PR", "LQRSV" are treated exceptionally, as special classes
"""

from copy import deepcopy
from pathlib import Path
from typing import List
from easydict import EasyDict as ED

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.databases.aux_data.cinc2020_aux_data import get_class_weight
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.utils import ecg_arrhythmia_knowledge as EAK
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths  # noqa: F401
from torch_ecg.model_configs import densenet_vanilla

__all__ = [
    "BaseCfg",
    "SpecialDetectorCfg",
    "TrainCfg",
    "ModelCfg",
]


print("-- Load configs for CPSC 2018")
_BASE_DIR = Path(__file__).parent.absolute()
_ONE_MINUTE_IN_MS = 60 * 1000


BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(parents=True, exist_ok=True)
BaseCfg.model_dir.mkdir(parents=True, exist_ok=True)
BaseCfg.fs = 500
BaseCfg.torch_dtype = DEFAULTS.torch_dtype
BaseCfg.n_leads = 12

SpecialDetectorCfg = CFG()
SpecialDetectorCfg.leads_ordering = deepcopy(EAK.Standard12Leads)
SpecialDetectorCfg.pr_fs_lower_bound = 47  # Hz
SpecialDetectorCfg.pr_spike_mph_ratio = (
    15  # ratio to the average amplitude of the signal
)
SpecialDetectorCfg.pr_spike_mpd = 300  # ms
SpecialDetectorCfg.pr_spike_prominence = 0.3
SpecialDetectorCfg.pr_spike_prominence_wlen = 120  # ms
SpecialDetectorCfg.pr_spike_inv_density_threshold = (
    2500  # inverse density (1/density), one spike per 2000 ms
)
SpecialDetectorCfg.pr_spike_leads_threshold = 7 / 12  # proportion
SpecialDetectorCfg.axis_qrs_mask_radius = 70  # ms
SpecialDetectorCfg.axis_method = "2-lead"  # can also be "3-lead"
SpecialDetectorCfg.brady_threshold = _ONE_MINUTE_IN_MS / 60  # ms, corr. to 60 bpm
SpecialDetectorCfg.tachy_threshold = _ONE_MINUTE_IN_MS / 100  # ms, corr. to 100 bpm
SpecialDetectorCfg.lqrsv_qrs_mask_radius = 60  # ms
SpecialDetectorCfg.lqrsv_ampl_bias = (
    0.02  # mV
)
SpecialDetectorCfg.lqrsv_ratio_threshold = 0.8
SpecialDetectorCfg.prwp_v3_thr = 0.3  # mV

# special classes using special detectors
_SPECIAL_CLASSES = ["Brady", "LAD", "RAD", "PR", "LQRSV"]


def _assign_classes(cfg: CFG, special_classes: List[str]) -> None:
    """ """
    cfg.special_classes = deepcopy(special_classes)
    cfg.tranche_class_weights = CFG(
        {
            t: get_class_weight(
                t,
                exclude_classes=cfg.special_classes,
                scored_only=True,
                threshold=20,
                min_weight=cfg.min_class_weight,
            )
            for t in [
                "A",
                # "B",
                # "AB",
                # "E",
                # "F",
            ]
        }
    )
    cfg.tranche_classes = CFG(
        {t: sorted(list(t_cw.keys())) for t, t_cw in cfg.tranche_class_weights.items()}
    )

    cfg.class_weights = get_class_weight(
        tranches="A",
        # tranches="ABEF",
        exclude_classes=cfg.special_classes,
        scored_only=True,
        threshold=20,
        min_weight=cfg.min_class_weight,
    )
    cfg.classes = sorted(list(cfg.class_weights.keys()))


# training configurations for machine learning and deep learning
TrainCfg = CFG()
TrainCfg.torch_dtype = BaseCfg.torch_dtype

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.model_dir = BaseCfg.model_dir
TrainCfg.final_model_name = None
TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
TrainCfg.keep_checkpoint_max = 5

TrainCfg.leads = deepcopy(EAK.Standard12Leads)

# configs of training data
TrainCfg.fs = BaseCfg.fs
TrainCfg.data_format = "channel_first"

TrainCfg.train_ratio = 0.8
TrainCfg.min_class_weight = 0.5
TrainCfg.tranches_for_training = ""  # one of "", "AB", "E", "F", "G"

# assign classes, class weights, tranche classes, etc.
# the no special classes version
_assign_classes(TrainCfg, [])

# configs of signal preprocessing
TrainCfg.normalize = CFG(
    method="z-score",
    mean=0.0,
    std=1.0,
)
# frequency band of the filter to apply, should be chosen very carefully
TrainCfg.bandpass = None

TrainCfg.label_smooth = False
TrainCfg.random_masking = False
TrainCfg.stretch_compress = False  # stretch or compress in time axis
TrainCfg.mixup = CFG(
    prob=0.6,
    alpha=0.3,
)

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 50
TrainCfg.batch_size = 64

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

TrainCfg.burn_in = 400
TrainCfg.steps = [5000, 10000]

TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 5

# configs of loss function
TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = (
    0.0  # flooding performed if positive, typically 0.45-0.55 for cinc2020?
)

TrainCfg.monitor = "accuracy"#"challenge_metric"

TrainCfg.log_step = 20
TrainCfg.eval_every = 20

# configs of model selection
TrainCfg.model_name = "cnn"  # "cnn", "cnn_phc"
TrainCfg.cnn_name = "multi_scopic"  # "multi_scopic", "resnetNS", "densenet_vanilla"
TrainCfg.rnn_name = "none"  # "none", "lstm"
TrainCfg.attn_name = "se"  # "none", "se", "gc", "nl"

# configs of inputs and outputs
# almost all records have duration >= 8s, most have duration >= 10s
# use `utils.utils_signal.ensure_siglen` to ensure signal length
TrainCfg.input_len = int(500 * 10.0)
# tolerance for records with length shorter than `TrainCfg.input_len`
TrainCfg.input_len_tol = int(0.2 * TrainCfg.input_len)
TrainCfg.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.siglen = TrainCfg.input_len

# constants for model inference
_bin_pred_thr = 0.5
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
_bin_pred_look_again_tol = 0.03
_bin_pred_nsr_thr = 0.1


TrainCfg.bin_pred_thr = _bin_pred_thr
TrainCfg.bin_pred_look_again_tol = _bin_pred_look_again_tol
TrainCfg.bin_pred_nsr_thr = _bin_pred_nsr_thr


# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
ModelCfg = CFG()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.bin_pred_thr = _bin_pred_thr
ModelCfg.bin_pred_look_again_tol = _bin_pred_look_again_tol
ModelCfg.bin_pred_nsr_thr = _bin_pred_nsr_thr

# the no special classes version
ModelCfg.special_classes = []
ModelCfg.dl_classes = deepcopy(TrainCfg.classes)
ModelCfg.tranche_classes = deepcopy(TrainCfg.tranche_classes)
ModelCfg.full_classes = ModelCfg.dl_classes + ModelCfg.special_classes

ModelCfg.dl_siglen = TrainCfg.siglen

ModelCfg.cnn_name = TrainCfg.cnn_name
ModelCfg.rnn_name = TrainCfg.rnn_name
ModelCfg.attn_name = TrainCfg.attn_name

ModelArchCfg = deepcopy(ECG_CRNN_CONFIG)
ModelArchCfg.cnn.multi_scopic_leadwise.block.batch_norm = "group_norm"  # False
_base_num_filters = 6
ModelArchCfg.cnn.multi_scopic.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]

ModelArchCfg.cnn.densenet_vanilla = deepcopy(densenet_vanilla)
ModelArchCfg.cnn.densenet_vanilla.init_num_filters = 72
ModelArchCfg.cnn.densenet_vanilla.growth_rates = 12
ModelArchCfg.cnn.resnetNS.stem.num_filters = 72

ModelArchCfg.clf = ED()
ModelArchCfg.clf.out_channels = [
    # not including the last linear layer, whose out channels equals n_classes
]
ModelArchCfg.clf.bias = True
ModelArchCfg.clf.dropouts = 0.0
ModelArchCfg.clf.activation = "mish"  # for a single layer `SeqLin`, activation is ignored

# model architectures configs
ModelCfg.update(ModelArchCfg)

