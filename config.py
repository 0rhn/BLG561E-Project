import os
from easydict import EasyDict as edict
from datetime import datetime


def create_config():
    """Initialize and return configuration dictionary."""
    config = edict()

    # Random seed for reproducibility
    config.SEED = 3035

    # Dataset configuration
    config.DATASET = "MovingDroneCrowd"
    config.NAME = ""
    config.encoder = "VGG16_FPN"

    # Training resume settings
    config.RESUME = False
    config.RESUME_PATH = ""
    config.PRE_TRAIN_COUNTER = ""

    # GPU configuration
    config.GPU_ID = "0,1,2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID

    # Cross-attention architecture parameters
    config.cross_attn_embed_dim = 256
    config.cross_attn_num_heads = 4
    config.mlp_ratio = 4
    config.cross_attn_depth = 2

    # Feature dimension
    config.FEATURE_DIM = 256

    # Optimizer settings
    config.LR_Base = 1e-5
    config.WEIGHT_DECAY = 1e-6

    # Training schedule
    config.MAX_EPOCH = 100
    config.VAL_INTERVAL = 10
    config.START_VAL = 20
    config.PRINT_FREQ = 20

    # Generate experiment name with timestamp
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    config.EXP_NAME = f"{timestamp}_{config.DATASET}_{config.LR_Base}_{config.NAME}"

    # Path configuration
    config.VAL_VIS_PATH = f"./exp/{config.DATASET}_val"
    config.EXP_PATH = os.path.join("./exp", config.DATASET)

    # Ensure experiment directory exists
    os.makedirs(config.EXP_PATH, exist_ok=True)

    return config
