"""
Configuration parameters for Score-Guided Manifold Projection (SGMP)
and baseline benchmarks. Optimized for NVIDIA A100 GPU on Google Colab and local environments.
"""

import os
import torch

class Config:
    # Hardware & Precision Acceleration
    SEED = 42
    DEVICE = os.environ.get("SGMP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True
    AMP_DTYPE = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
    ALLOW_TF32 = True
    CUDNN_BENCHMARK = True
    NUM_WORKERS = int(os.environ.get("SGMP_NUM_WORKERS", 4 if torch.cuda.is_available() else 0))
    PIN_MEMORY = torch.cuda.is_available()
    
    # Dataset Parameters
    DATASET_NAME = "FashionMNIST"  # Primary visual pattern benchmark (60,000 train / 10,000 test)
    NUM_CLASSES = 10
    IN_CHANNELS = 1
    IMAGE_SIZE = 28
    BATCH_SIZE = int(os.environ.get("SGMP_BATCH_SIZE", 256 if torch.cuda.is_available() else 64))
    
    # Dataset Subsampling (None uses the full standard dataset)
    FULL_DATASET = os.environ.get("SGMP_FULL_DATASET", "1" if torch.cuda.is_available() else "0") == "1"
    TRAIN_SAMPLES = None if FULL_DATASET else 640
    VAL_SAMPLES = None if FULL_DATASET else 200
    TEST_SAMPLES = None if FULL_DATASET else 200
    
    # Training Parameters
    EPOCHS = int(os.environ.get("SGMP_EPOCHS", 25 if torch.cuda.is_available() else 3))
    EPOCHS_SCORE_NET = int(os.environ.get("SGMP_EPOCHS_SCORE", 60 if torch.cuda.is_available() else 3))
    EPOCHS_CLASSIFIER = int(os.environ.get("SGMP_EPOCHS_CLF", 25 if torch.cuda.is_available() else 3))
    EPOCHS_ADV = int(os.environ.get("SGMP_EPOCHS_ADV", 20 if torch.cuda.is_available() else 3))
    EPOCHS_DAE = int(os.environ.get("SGMP_EPOCHS_DAE", 20 if torch.cuda.is_available() else 3))
    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    COSINE_LR_MIN = 1e-5
    
    # Exponential Moving Average (EMA) for ScoreNet stability
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # Diffusion / Score Model Parameters (Karras EDM Formulation)
    DIFFUSION_TIME_STEPS = 1000
    SIGMA_MIN = 0.01
    SIGMA_MAX = 1.0
    SIGMA_DATA = 0.25              # Expected standard deviation of data distribution
    PURIFY_NOISE_LEVEL = 0.25      # Reference forward noise level
    
    # Adaptive Purification Parameters
    ADAPTIVE_PURIFY = True         # Modulate noise injection by corruption estimate & confidence
    MIN_PURIFY_SIGMA = 0.02        # Conservative noise floor for clean patterns
    MAX_PURIFY_SIGMA = 0.35        # Maximum noise ceiling for severe corruptions
    
    # SGMP Framework Hyperparameters (Our Method)
    SGMP_STEPS = 4                 # Amortized Trajectory Fast Solver steps (default)
    DIFFPURE_STEPS = 12            # Baseline DiffPure steps
    GUIDANCE_SCALE = 2.0           # CMLB likelihood guidance weight (increased for stronger class-aware purification)
    CONFIDENCE_EXPONENT = 1.5      # Epistemic certainty modulation exponent (sharper decay)
    SMC_CONTRACTION_WEIGHT = 0.30  # Spectral Manifold Contraction strength (reduced to avoid harming accuracy)
    SMC_COSINE_THRESHOLD = -0.25   # Only contract when cos(score, grad) is below this (strongly antagonistic)
    RESIDUAL_CONSISTENCY = 0.15    # Data fidelity weight for trajectory refinement (reduced to avoid re-injecting corruption)
    RESIDUAL_CONSISTENCY_CLEAN = 0.40  # Higher consistency for near-clean inputs
    CORRECTOR_STEPS = 1            # Corrector Langevin steps per interval
    CORRECTOR_SNR = 0.15           # Corrector Signal-to-Noise ratio
    CORRECTOR_ENABLED = True       # Enable true Langevin corrector step
    
    # Adversarial-adaptive mode
    ADVERSARIAL_STEPS = 6          # More steps when adversarial input is detected
    ADVERSARIAL_SIGMA = 0.40       # Larger noise for adversarial purification
    
    # Benchmark Corruptions & Perturbations
    SEVERITY_LEVELS = [1, 2, 3, 4, 5]
    CORRUPTION_TYPES = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "motion_blur",
        "fog",
        "frost",
        "contrast",
        "pixelate",
        "occlusion",
        "cutout",
        "adversarial_pgd"
    ]
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    FIGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Paper", "figs"))

    @classmethod
    def apply_profile(cls, profile="a100"):
        """Select execution preset: 'a100', 'colab', 'cpu', or 'smoke_test'."""
        if profile in ("a100", "colab"):
            cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            cls.USE_AMP = True
            cls.AMP_DTYPE = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
            cls.ALLOW_TF32 = True
            cls.CUDNN_BENCHMARK = True
            cls.NUM_WORKERS = 4
            cls.PIN_MEMORY = True
            cls.BATCH_SIZE = 256
            cls.FULL_DATASET = True
            cls.TRAIN_SAMPLES = None
            cls.VAL_SAMPLES = None
            cls.TEST_SAMPLES = None
            cls.EPOCHS_CLASSIFIER = 25
            cls.EPOCHS_SCORE_NET = 60
            cls.EPOCHS_ADV = 20
            cls.EPOCHS_DAE = 20
            cls.USE_EMA = True
        elif profile == "smoke_test":
            cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            cls.BATCH_SIZE = 64
            cls.FULL_DATASET = False
            cls.TRAIN_SAMPLES = 256
            cls.VAL_SAMPLES = 128
            cls.TEST_SAMPLES = 128
            cls.EPOCHS_CLASSIFIER = 1
            cls.EPOCHS_SCORE_NET = 1
            cls.EPOCHS_ADV = 1
            cls.EPOCHS_DAE = 1
            cls.NUM_WORKERS = 0
        elif profile == "cpu":
            cls.DEVICE = "cpu"
            cls.USE_AMP = False
            cls.NUM_WORKERS = 0
            cls.PIN_MEMORY = False
            cls.BATCH_SIZE = 64
            cls.FULL_DATASET = False
            cls.TRAIN_SAMPLES = 640
            cls.VAL_SAMPLES = 200
            cls.TEST_SAMPLES = 200
            cls.EPOCHS_CLASSIFIER = 3
            cls.EPOCHS_SCORE_NET = 3
            cls.EPOCHS_ADV = 3
            cls.EPOCHS_DAE = 3

os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
os.makedirs(Config.FIGS_DIR, exist_ok=True)

if Config.DEVICE == "cuda" and Config.ALLOW_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if Config.DEVICE == "cuda" and Config.CUDNN_BENCHMARK:
    torch.backends.cudnn.benchmark = True

