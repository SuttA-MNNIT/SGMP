"""
Standardized Corruption Benchmark for Pattern Recognition.
Implements corruptions inspired by Hendrycks & Dietterich (ICLR 2019) benchmark suites:
Gaussian, shot, impulse noise, defocus blur, motion blur, fog, frost, contrast,
pixelate, spatial occlusion, cutout, and gradient-based adversarial perturbations (PGD).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import os
from config import Config

def add_gaussian_noise(x, severity=3):
    c = [0.08, 0.15, 0.25, 0.38, 0.52][severity - 1]
    noise = np.random.normal(0, c, x.shape)
    return np.clip(x + noise, 0.0, 1.0)

def add_shot_noise(x, severity=3):
    c = [60, 25, 12, 5, 2][severity - 1]
    # Poisson noise
    x_scaled = x * c
    noisy = np.random.poisson(np.maximum(0, x_scaled)) / c
    return np.clip(noisy, 0.0, 1.0)

def add_impulse_noise(x, severity=3):
    c = [0.03, 0.06, 0.12, 0.22, 0.35][severity - 1]
    res = x.copy()
    mask = np.random.uniform(0, 1, x.shape) < c
    salt = np.random.uniform(0, 1, x.shape) < 0.5
    res[mask & salt] = 1.0
    res[mask & (~salt)] = 0.0
    return res

def add_defocus_blur(x, severity=3):
    sigma = [0.8, 1.2, 1.8, 2.5, 3.5][severity - 1]
    res = np.zeros_like(x)
    for b in range(x.shape[0]):
        for ch in range(x.shape[1]):
            res[b, ch] = gaussian_filter(x[b, ch], sigma=sigma)
    return np.clip(res, 0.0, 1.0)

def add_motion_blur(x, severity=3):
    length = [3, 5, 7, 9, 11][severity - 1]
    kernel = np.zeros((length, length))
    kernel[length // 2, :] = 1.0 / length
    from scipy.signal import convolve2d
    res = np.zeros_like(x)
    for b in range(x.shape[0]):
        for ch in range(x.shape[1]):
            res[b, ch] = convolve2d(x[b, ch], kernel, mode='same', boundary='symm')
    return np.clip(res, 0.0, 1.0)

def add_fog(x, severity=3):
    c = [0.15, 0.28, 0.42, 0.58, 0.72][severity - 1]
    h, w = x.shape[2], x.shape[3]
    fog_layer = np.linspace(0.4, 0.8, h)[:, None] * np.linspace(0.4, 0.8, w)[None, :]
    fog_layer = fog_layer[None, None, :, :]
    res = (1 - c) * x + c * fog_layer
    return np.clip(res, 0.0, 1.0)

def add_frost(x, severity=3):
    c = [0.15, 0.28, 0.40, 0.55, 0.70][severity - 1]
    noise = np.random.laplace(0, 0.2, x.shape)
    return np.clip((1 - c) * x + c * noise + 0.1 * c, 0.0, 1.0)

def add_contrast(x, severity=3):
    c = [0.75, 0.55, 0.38, 0.22, 0.12][severity - 1]
    means = np.mean(x, axis=(2, 3), keepdims=True)
    return np.clip((x - means) * c + means, 0.0, 1.0)

def add_pixelate(x, severity=3):
    factor = [0.7, 0.5, 0.35, 0.25, 0.18][severity - 1]
    import cv2
    res = np.zeros_like(x)
    h, w = x.shape[2], x.shape[3]
    for b in range(x.shape[0]):
        for ch in range(x.shape[1]):
            img = x[b, ch]
            new_h, new_w = max(2, int(h * factor)), max(2, int(w * factor))
            down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
            res[b, ch] = up
    return np.clip(res, 0.0, 1.0)

def add_occlusion(x, severity=3):
    block_size = [4, 7, 10, 13, 16][severity - 1]
    res = x.copy()
    h, w = x.shape[2], x.shape[3]
    for b in range(x.shape[0]):
        top = np.random.randint(0, max(1, h - block_size))
        left = np.random.randint(0, max(1, w - block_size))
        res[b, :, top:top+block_size, left:left+block_size] = 0.0
    return res

def add_cutout(x, severity=3):
    num_cuts = [1, 2, 3, 4, 5][severity - 1]
    res = x.copy()
    h, w = x.shape[2], x.shape[3]
    for b in range(x.shape[0]):
        for _ in range(num_cuts):
            ch = np.random.randint(4, 8)
            cw = np.random.randint(4, 8)
            top = np.random.randint(0, max(1, h - ch))
            left = np.random.randint(0, max(1, w - cw))
            res[b, :, top:top+ch, left:left+cw] = np.random.uniform(0, 1)
    return res

def apply_corruption(x_tensor, corruption_name, severity=3):
    """
    Applies designated corruption to a PyTorch tensor (B, C, H, W).
    """
    if corruption_name == "clean":
        return x_tensor
    x_np = x_tensor.detach().cpu().numpy()
    if corruption_name == "gaussian_noise":
        corrupted = add_gaussian_noise(x_np, severity)
    elif corruption_name == "shot_noise":
        corrupted = add_shot_noise(x_np, severity)
    elif corruption_name == "impulse_noise":
        corrupted = add_impulse_noise(x_np, severity)
    elif corruption_name == "defocus_blur":
        corrupted = add_defocus_blur(x_np, severity)
    elif corruption_name == "motion_blur":
        corrupted = add_motion_blur(x_np, severity)
    elif corruption_name == "fog":
        corrupted = add_fog(x_np, severity)
    elif corruption_name == "frost":
        corrupted = add_frost(x_np, severity)
    elif corruption_name == "contrast":
        corrupted = add_contrast(x_np, severity)
    elif corruption_name == "pixelate":
        corrupted = add_pixelate(x_np, severity)
    elif corruption_name == "occlusion":
        corrupted = add_occlusion(x_np, severity)
    elif corruption_name == "cutout":
        corrupted = add_cutout(x_np, severity)
    elif corruption_name == "adversarial_pgd":
        # Handled dynamically via model gradients in evaluate.py
        return x_tensor
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    return torch.tensor(corrupted, dtype=torch.float32, device=x_tensor.device)


class SyntheticPatternDataset(Dataset):
    """
    Fallback deterministic multi-class geometric & visual pattern recognition dataset.
    Generates rich 10-class patterns (lines, concentric circles, chevrons, crosses,
    checkerboards, rings, spirals, sine waves, diagonal lattices, radial stars).
    """
    def __init__(self, num_samples=1000, seed=42):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        h, w = Config.IMAGE_SIZE, Config.IMAGE_SIZE
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        center_y, center_x = h / 2.0, w / 2.0
        r = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        theta = np.arctan2(y_coords - center_y, x_coords - center_x)

        for i in range(num_samples):
            cls = i % Config.NUM_CLASSES
            img = np.zeros((h, w), dtype=np.float32)
            noise = np.random.normal(0, 0.05, (h, w))
            
            if cls == 0:  # Concentric rings
                freq = np.random.uniform(0.6, 0.8)
                img = 0.5 + 0.5 * np.sin(freq * r)
            elif cls == 1:  # Radial star rays
                rays = 6
                img = 0.5 + 0.5 * np.cos(rays * theta)
            elif cls == 2:  # Horizontal & vertical checkerboard
                freq = 0.45
                img = 0.5 + 0.5 * np.sign(np.sin(freq * x_coords) * np.sin(freq * y_coords))
            elif cls == 3:  # Diagonal waves
                freq = 0.35
                img = 0.5 + 0.5 * np.sin(freq * (x_coords + y_coords))
            elif cls == 4:  # Concentric squares
                dist_l1 = np.maximum(np.abs(x_coords - center_x), np.abs(y_coords - center_y))
                img = 0.5 + 0.5 * np.cos(0.7 * dist_l1)
            elif cls == 5:  # Archimedean spiral
                img = 0.5 + 0.5 * np.sin(r * 0.7 - theta * 2)
            elif cls == 6:  # Cross lattice
                img = np.exp(-((x_coords - center_x)**2) / 12.0) + np.exp(-((y_coords - center_y)**2) / 12.0)
            elif cls == 7:  # Circular disc / bubble
                img = (r < (Config.IMAGE_SIZE / 3.0)).astype(np.float32)
            elif cls == 8:  # Gabor-like directional ripple
                img = 0.5 + 0.5 * np.sin(0.5 * (x_coords * 0.8 - y_coords * 0.6)) * np.exp(-r**2 / 120.0)
            elif cls == 9:  # Corner quadrants
                img = ((x_coords > center_x) ^ (y_coords > center_y)).astype(np.float32)
                
            img = np.clip(img + noise, 0.0, 1.0)
            self.data.append(img[None, :, :])
            self.labels.append(cls)
            
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloaders(batch_size=None, num_workers=None, pin_memory=None, full_dataset=None):
    """
    Returns train and test dataloaders optimized for high-throughput GPU training.
    Attempts to download/load FashionMNIST; falls back to SyntheticPatternDataset if offline.
    """
    data_dir = os.path.join(Config.BASE_DIR, "data", "cache")
    os.makedirs(data_dir, exist_ok=True)
    
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS
    pm = pin_memory if pin_memory is not None else Config.PIN_MEMORY
    full = full_dataset if full_dataset is not None else Config.FULL_DATASET
    
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_full = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_full = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
        
        if full or Config.TRAIN_SAMPLES is None:
            train_set = train_full
            test_set = test_full
            print(f"Loaded full Fashion-MNIST benchmark dataset ({len(train_set)} train / {len(test_set)} test samples).")
        else:
            num_train = min(Config.TRAIN_SAMPLES, len(train_full))
            num_test = min(Config.TEST_SAMPLES, len(test_full))
            train_set = torch.utils.data.Subset(train_full, list(range(num_train)))
            test_set = torch.utils.data.Subset(test_full, list(range(num_test)))
            print(f"Loaded subsampled Fashion-MNIST ({len(train_set)} train / {len(test_set)} test samples).")
        
        loader_kwargs = {
            "batch_size": bs,
            "pin_memory": pm,
            "num_workers": nw,
            "persistent_workers": (nw > 0)
        }
        train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
        return train_loader, test_loader
    except Exception as e:
        print(f"FashionMNIST download/load unavailable ({e}), utilizing synthetic pattern dataset.")
        train_n = 10000 if (full or Config.TRAIN_SAMPLES is None) else Config.TRAIN_SAMPLES
        test_n = 2000 if (full or Config.TEST_SAMPLES is None) else Config.TEST_SAMPLES
        train_set = SyntheticPatternDataset(num_samples=train_n, seed=Config.SEED)
        test_set = SyntheticPatternDataset(num_samples=test_n, seed=Config.SEED + 100)
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
        return train_loader, test_loader

