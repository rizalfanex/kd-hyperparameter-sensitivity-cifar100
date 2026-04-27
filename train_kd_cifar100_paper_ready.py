import os
import time
import json
import copy
import random
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models


# =========================================================
# 1. CONFIG
# =========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs_kd_paper_ready"
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATA_ROOT = "./data"

NUM_CLASSES = 100
IMG_SIZE = 128
BATCH_SIZE = 128

# Kalau mau lebih cepat, ubah ke 30
EPOCHS_TEACHER = 40
EPOCHS_STUDENT = 40

# WINDOWS SAFE
NUM_WORKERS = 0

LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

KD_ALPHA = 0.5
KD_TEMPERATURE = 4.0

USE_PRETRAINED = True
USE_AMP = True

VAL_RATIO = 0.1  # 10% dari CIFAR-100 train jadi validation

TEACHER_MODEL = "resnet50"
STUDENT_MODELS = [
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
]

# Ablation study config
ABLATION_EPOCHS = 30
ABLATION_TEMPERATURES = [2, 4, 6]
ABLATION_ALPHAS = [0.3, 0.5, 0.7]
ABLATION_FIXED_ALPHA = 0.5
ABLATION_FIXED_TEMPERATURE = 4.0
ABLATION_STUDENT = "mobilenet_v2"
ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")
os.makedirs(ABLATION_DIR, exist_ok=True)


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


set_seed(SEED)


# =========================================================
# 3. DATASET WRAPPER (TOP-LEVEL, WINDOWS SAFE)
# =========================================================
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, target = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# =========================================================
# 4. DATA
# =========================================================
def get_transforms(img_size=128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.06
        ),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
    return train_transform, eval_transform


def get_dataloaders(
    data_root="./data",
    batch_size=128,
    img_size=128,
    num_workers=0,
    val_ratio=0.1
):
    train_tf, eval_tf = get_transforms(img_size)

    full_train_for_split = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=None
    )

    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=eval_tf
    )

    total_len = len(full_train_for_split)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_subset_raw, val_subset_raw = random_split(
        full_train_for_split,
        [train_len, val_len],
        generator=generator
    )

    train_set = TransformSubset(train_subset_raw, train_tf)
    val_set = TransformSubset(val_subset_raw, eval_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, train_len, val_len, len(test_set)


# =========================================================
# 5. MODEL FACTORY
# =========================================================
def create_model(model_name: str, num_classes: int = 100, pretrained: bool = True):
    weights = None
    if pretrained:
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        elif model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        elif model_name == "shufflenet_v2_x1_0":
            weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1

    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 6. METRICS
# =========================================================
def accuracy_topk(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append((correct_k * 100.0 / batch_size).item())
        return results


# =========================================================
# 7. KNOWLEDGE DISTILLATION LOSS
# =========================================================
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets):
        ce_loss = self.ce(student_logits, targets)

        T = self.temperature
        log_p_student = F.log_softmax(student_logits / T, dim=1)
        p_teacher = F.softmax(teacher_logits / T, dim=1)
        kd_loss = self.kl(log_p_student, p_teacher) * (T * T)

        total_loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
        return total_loss, ce_loss.detach(), kd_loss.detach()


# =========================================================
# 8. LOG STRUCTURE
# =========================================================
@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_top1: float
    train_top5: float
    val_loss: float
    val_top1: float
    val_top5: float
    lr: float


# =========================================================
# 9. AMP HELPERS
# =========================================================
def autocast_context(device):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(USE_AMP and device == "cuda")
    )


def build_grad_scaler(device):
    if device == "cuda":
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    return torch.amp.GradScaler(enabled=False)


# =========================================================
# 10. TRAIN / EVAL
# =========================================================
def train_one_epoch_standard(model, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))
        bs = images.size(0)

        running_loss += loss.item() * bs
        running_top1 += top1 * bs
        running_top5 += top5 * bs
        total += bs

        pbar.set_postfix(
            loss=f"{running_loss/total:.4f}",
            top1=f"{running_top1/total:.2f}"
        )

    return running_loss / total, running_top1 / total, running_top5 / total


def train_one_epoch_kd(student, teacher, loader, optimizer, scaler, kd_criterion, device):
    student.train()
    teacher.eval()

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total = 0

    pbar = tqdm(loader, desc="Train KD", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with autocast_context(device):
                teacher_logits = teacher(images)

        with autocast_context(device):
            student_logits = student(images)
            loss, _, _ = kd_criterion(student_logits, teacher_logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        top1, top5 = accuracy_topk(student_logits, targets, topk=(1, 5))
        bs = images.size(0)

        running_loss += loss.item() * bs
        running_top1 += top1 * bs
        running_top5 += top5 * bs
        total += bs

        pbar.set_postfix(
            loss=f"{running_loss/total:.4f}",
            top1=f"{running_top1/total:.2f}"
        )

    return running_loss / total, running_top1 / total, running_top5 / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total = 0

    pbar = tqdm(loader, desc="Eval", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device):
            outputs = model(images)
            loss = criterion(outputs, targets)

        top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))
        bs = images.size(0)

        running_loss += loss.item() * bs
        running_top1 += top1 * bs
        running_top5 += top5 * bs
        total += bs

        pbar.set_postfix(
            loss=f"{running_loss/total:.4f}",
            top1=f"{running_top1/total:.2f}"
        )

    return running_loss / total, running_top1 / total, running_top5 / total


# =========================================================
# 11. SAVE / LATENCY
# =========================================================
def save_history_json(history, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in history], f, indent=2)


@torch.no_grad()
def measure_latency(model, device="cuda", img_size=128, warmup=30, runs=100):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size).to(device)

    if device == "cuda":
        torch.cuda.empty_cache()

    for _ in range(warmup):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return float(np.mean(times))


# =========================================================
# 12. TRAINING WRAPPER
# =========================================================
def train_standard_model(model_name, train_loader, val_loader, test_loader, epochs, device):
    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=USE_PRETRAINED
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )
    scaler = build_grad_scaler(device)

    best_val_top1 = -1.0
    best_state = None
    history = []

    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_standard_best.pt")
    log_path = os.path.join(LOG_DIR, f"{model_name}_standard_history.json")

    print(f"\n[INFO] Training STANDARD model: {model_name}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}] - {model_name} STANDARD")

        train_loss, train_top1, train_top5 = train_one_epoch_standard(
            model, train_loader, optimizer, scaler, device
        )
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append(EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            train_top1=train_top1,
            train_top5=train_top5,
            val_loss=val_loss,
            val_top1=val_top1,
            val_top5=val_top5,
            lr=current_lr
        ))

        print(
            f"[{model_name} STANDARD] "
            f"train_loss={train_loss:.4f}, train_top1={train_top1:.2f}, train_top5={train_top5:.2f} | "
            f"val_loss={val_loss:.4f}, val_top1={val_top1:.2f}, val_top5={val_top5:.2f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_name": model_name,
                "training_type": "standard",
                "best_val_top1": best_val_top1,
                "state_dict": best_state,
            }, ckpt_path)

    total_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    save_history_json(history, log_path)

    params_m = count_parameters(model) / 1e6
    latency_ms = measure_latency(model, device=device)

    test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)

    result = {
        "model": model_name,
        "training_type": "standard",
        "params_m": params_m,
        "best_val_top1": best_val_top1,
        "test_loss": test_loss,
        "top1": test_top1,
        "top5": test_top5,
        "latency_ms": latency_ms,
        "train_time_sec": total_time,
        "checkpoint": ckpt_path,
        "history_path": log_path,
    }
    return model, history, result


def train_kd_student(student_name, teacher_model, train_loader, val_loader, test_loader, epochs, device):
    student = create_model(
        model_name=student_name,
        num_classes=NUM_CLASSES,
        pretrained=USE_PRETRAINED
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )
    scaler = build_grad_scaler(device)
    kd_criterion = KnowledgeDistillationLoss(
        alpha=KD_ALPHA,
        temperature=KD_TEMPERATURE
    )

    best_val_top1 = -1.0
    best_state = None
    history = []

    ckpt_path = os.path.join(CKPT_DIR, f"{student_name}_kd_best.pt")
    log_path = os.path.join(LOG_DIR, f"{student_name}_kd_history.json")

    print(f"\n[INFO] Training KD model: {student_name}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}] - {student_name} KD")

        train_loss, train_top1, train_top5 = train_one_epoch_kd(
            student, teacher_model, train_loader, optimizer, scaler, kd_criterion, device
        )
        val_loss, val_top1, val_top5 = evaluate(student, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append(EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            train_top1=train_top1,
            train_top5=train_top5,
            val_loss=val_loss,
            val_top1=val_top1,
            val_top5=val_top5,
            lr=current_lr
        ))

        print(
            f"[{student_name} KD] "
            f"train_loss={train_loss:.4f}, train_top1={train_top1:.2f}, train_top5={train_top5:.2f} | "
            f"val_loss={val_loss:.4f}, val_top1={val_top1:.2f}, val_top5={val_top5:.2f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = copy.deepcopy(student.state_dict())
            torch.save({
                "model_name": student_name,
                "training_type": "kd",
                "best_val_top1": best_val_top1,
                "alpha": KD_ALPHA,
                "temperature": KD_TEMPERATURE,
                "state_dict": best_state,
            }, ckpt_path)

    total_time = time.time() - start_time

    if best_state is not None:
        student.load_state_dict(best_state)

    save_history_json(history, log_path)

    params_m = count_parameters(student) / 1e6
    latency_ms = measure_latency(student, device=device)

    test_loss, test_top1, test_top5 = evaluate(student, test_loader, device)

    result = {
        "model": student_name,
        "training_type": "kd",
        "params_m": params_m,
        "best_val_top1": best_val_top1,
        "test_loss": test_loss,
        "top1": test_top1,
        "top5": test_top5,
        "latency_ms": latency_ms,
        "train_time_sec": total_time,
        "checkpoint": ckpt_path,
        "history_path": log_path,
    }
    return student, history, result


# =========================================================
# 12b. ABLATION STUDY RUNNER
# =========================================================
def run_kd_ablation(
    student_name,
    teacher_model,
    train_loader,
    val_loader,
    test_loader,
    param_type,
    param_values,
    fixed_alpha=0.5,
    fixed_temperature=4.0,
    epochs=30,
    device="cuda",
):
    """
    Run KD ablation over a list of hyperparameter values.

    Args:
        param_type: "temperature" or "alpha"
        param_values: list of values to sweep
        fixed_alpha: alpha used when sweeping temperature
        fixed_temperature: temperature used when sweeping alpha
    Returns:
        List of result dicts, one per param value.
    """
    ablation_results = []

    for val in param_values:
        if param_type == "temperature":
            alpha, temperature = fixed_alpha, val
        elif param_type == "alpha":
            alpha, temperature = val, fixed_temperature
        else:
            raise ValueError(f"Unknown param_type: {param_type}")

        tag = f"{student_name}_ablation_{param_type}_{val}"
        ckpt_path = os.path.join(ABLATION_DIR, f"{tag}_best.pt")
        log_path = os.path.join(ABLATION_DIR, f"{tag}_history.json")

        # --- RESUME: skip if checkpoint already exists ---
        if os.path.exists(ckpt_path) and os.path.exists(log_path):
            print(f"\n{'=' * 70}")
            print(f"[SKIP] {param_type}={val} | checkpoint found, loading results...")
            print(f"{'=' * 70}")

            student = create_model(student_name, NUM_CLASSES, pretrained=False).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            student.load_state_dict(ckpt["state_dict"])
            student.eval()

            latency_ms = measure_latency(student, device=device)
            test_loss, test_top1, test_top5 = evaluate(student, test_loader, device)

            result = {
                "experiment_type": param_type,
                "parameter_value": val,
                "alpha": alpha,
                "temperature": temperature,
                "best_val_top1": round(float(ckpt["best_val_top1"]), 2),
                "top1": round(test_top1, 2),
                "top5": round(test_top5, 2),
                "latency_ms": round(latency_ms, 4),
                "train_time_sec": 0.0,
            }
            ablation_results.append(result)
            print(
                f"[LOADED] {param_type}={val} => "
                f"test_top1={test_top1:.2f}, test_top5={test_top5:.2f}, "
                f"best_val_top1={ckpt['best_val_top1']:.2f}"
            )
            continue

        print(f"\n{'=' * 70}")
        print(f"[ABLATION] {param_type}={val} | alpha={alpha}, T={temperature} | epochs={epochs}")
        print(f"{'=' * 70}")

        set_seed(SEED)

        student = create_model(
            model_name=student_name,
            num_classes=NUM_CLASSES,
            pretrained=USE_PRETRAINED,
        ).to(device)

        optimizer = torch.optim.AdamW(
            student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = build_grad_scaler(device)
        kd_criterion = KnowledgeDistillationLoss(alpha=alpha, temperature=temperature)

        best_val_top1 = -1.0
        best_state = None
        history = []

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_top1, train_top5 = train_one_epoch_kd(
                student, teacher_model, train_loader, optimizer, scaler, kd_criterion, device
            )
            val_loss, val_top1, val_top5 = evaluate(student, val_loader, device)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            history.append(EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                train_top1=train_top1,
                train_top5=train_top5,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                lr=current_lr,
            ))

            print(
                f"  Epoch [{epoch}/{epochs}] "
                f"train_loss={train_loss:.4f}, train_top1={train_top1:.2f} | "
                f"val_loss={val_loss:.4f}, val_top1={val_top1:.2f}"
            )

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_state = copy.deepcopy(student.state_dict())

        total_time = time.time() - start_time

        if best_state is not None:
            student.load_state_dict(best_state)
            torch.save({
                "model_name": student_name,
                "training_type": f"ablation_{param_type}",
                "param_value": val,
                "alpha": alpha,
                "temperature": temperature,
                "best_val_top1": best_val_top1,
                "state_dict": best_state,
            }, ckpt_path)

        save_history_json(history, log_path)

        latency_ms = measure_latency(student, device=device)
        test_loss, test_top1, test_top5 = evaluate(student, test_loader, device)

        result = {
            "experiment_type": param_type,
            "parameter_value": val,
            "alpha": alpha,
            "temperature": temperature,
            "best_val_top1": round(best_val_top1, 2),
            "top1": round(test_top1, 2),
            "top5": round(test_top5, 2),
            "latency_ms": round(latency_ms, 4),
            "train_time_sec": round(total_time, 2),
        }
        ablation_results.append(result)

        print(
            f"[ABLATION RESULT] {param_type}={val} => "
            f"test_top1={test_top1:.2f}, test_top5={test_top5:.2f}, "
            f"best_val_top1={best_val_top1:.2f}, time={total_time:.1f}s"
        )

    return ablation_results


# =========================================================
# 13. PLOTTING — Q1 JOURNAL STYLE
# =========================================================
def _apply_q1_style():
    """Apply Q1-journal rcParams. Call matplotlib.rcdefaults() after plt.close()."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _reset_style():
    import matplotlib
    matplotlib.rcdefaults()


# ---- pretty model names for legends / axes ----
_PRETTY = {
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNetV2",
    "shufflenet_v2_x1_0": "ShuffleNetV2",
}
_TYPE_PRETTY = {"standard": "Baseline", "kd": "KD", "teacher": "Teacher"}

_COLORS = {
    "standard": "#1f77b4",
    "kd": "#d62728",
    "teacher": "#2ca02c",
}
_MARKERS = {"standard": "o", "kd": "s", "teacher": "^"}


def plot_training_curve(student_name, history_std, history_kd, save_path):
    _apply_q1_style()
    pretty = _PRETTY.get(student_name, student_name)

    epochs_std = [x.epoch for x in history_std]
    epochs_kd = [x.epoch for x in history_kd]

    std_train = [x.train_top1 for x in history_std]
    std_val = [x.val_top1 for x in history_std]
    kd_train = [x.train_top1 for x in history_kd]
    kd_val = [x.val_top1 for x in history_kd]

    fig, ax = plt.subplots(figsize=(5, 3.8))

    ax.plot(epochs_std, std_train, linewidth=1.4, color="#1f77b4", linestyle="-",
            label="Baseline – Train")
    ax.plot(epochs_std, std_val, linewidth=1.4, color="#1f77b4", linestyle="--",
            label="Baseline – Val")
    ax.plot(epochs_kd, kd_train, linewidth=1.4, color="#d62728", linestyle="-",
            label="KD – Train")
    ax.plot(epochs_kd, kd_val, linewidth=1.4, color="#d62728", linestyle="--",
            label="KD – Val")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(loc="lower right", frameon=True, fontsize=7.5, edgecolor="0.6",
              fancybox=False, framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    _reset_style()


def plot_bar_top1(results_df, save_path):
    _apply_q1_style()

    labels = [
        f"{_PRETTY.get(r['model'], r['model'])}\n({_TYPE_PRETTY.get(r['training_type'], r['training_type'])})"
        for _, r in results_df.iterrows()
    ]
    values = results_df["top1"].values
    colors = [_COLORS.get(r["training_type"], "#999") for _, r in results_df.iterrows()]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(range(len(values)), values, width=0.55, color=colors,
                  edgecolor="black", linewidth=0.6)

    for i, v in enumerate(values):
        ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=7.5)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    _reset_style()


def plot_scatter_accuracy_vs_params(results_df, save_path):
    _apply_q1_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.8))

    for _, row in results_df.iterrows():
        tt = row["training_type"]
        marker = _MARKERS.get(tt, "o")
        color = _COLORS.get(tt, "#999")
        pretty = _PRETTY.get(row["model"], row["model"])
        label = f"{pretty} ({_TYPE_PRETTY.get(tt, tt)})"
        ax.scatter(row["params_m"], row["top1"], marker=marker, s=90,
                   color=color, edgecolors="black", linewidths=0.5,
                   label=label, zorder=3)
        ax.annotate(pretty, (row["params_m"], row["top1"]),
                    textcoords="offset points", xytext=(5, 6),
                    fontsize=7, color="0.25")

    ax.set_xlabel("Parameters (M)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=7, edgecolor="0.6",
              fancybox=False, framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    _reset_style()


def plot_scatter_accuracy_vs_latency(results_df, save_path):
    _apply_q1_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.8))

    for _, row in results_df.iterrows():
        tt = row["training_type"]
        marker = _MARKERS.get(tt, "o")
        color = _COLORS.get(tt, "#999")
        pretty = _PRETTY.get(row["model"], row["model"])
        label = f"{pretty} ({_TYPE_PRETTY.get(tt, tt)})"
        ax.scatter(row["latency_ms"], row["top1"], marker=marker, s=90,
                   color=color, edgecolors="black", linewidths=0.5,
                   label=label, zorder=3)
        ax.annotate(pretty, (row["latency_ms"], row["top1"]),
                    textcoords="offset points", xytext=(5, 6),
                    fontsize=7, color="0.25")

    ax.set_xlabel("Inference Latency (ms)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=7, edgecolor="0.6",
              fancybox=False, framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    _reset_style()


def plot_kd_gain(gain_df, save_path):
    _apply_q1_style()
    labels = [_PRETTY.get(s, s) for s in gain_df["student"].tolist()]
    gains = gain_df["top1_gain"].tolist()

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    bars = ax.bar(range(len(gains)), gains, width=0.45, color="#d62728",
                  edgecolor="black", linewidth=0.6)

    for i, g in enumerate(gains):
        ax.text(i, g + 0.04, f"+{g:.2f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Top-1 Gain (%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_ylim(0, max(gains) * 1.35)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    _reset_style()


def plot_ablation_curve(ablation_df, param_type, save_path):
    """Q1-journal-quality line plot: Top-1 accuracy vs ablation parameter."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    subset = ablation_df[ablation_df["experiment_type"] == param_type].copy()
    subset = subset.sort_values("parameter_value")

    x = subset["parameter_value"].values
    y_top1 = subset["top1"].values
    y_top5 = subset["top5"].values

    if param_type == "temperature":
        xlabel = "Temperature ($T$)"
        fixed_label = f"$\\alpha = {subset['alpha'].iloc[0]}$"
    else:
        xlabel = "Distillation Weight ($\\alpha$)"
        fixed_label = f"$T = {int(subset['temperature'].iloc[0])}$"

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.plot(x, y_top1, marker="o", linewidth=1.8, markersize=7,
            color="#1f77b4", markeredgecolor="white", markeredgewidth=0.8,
            label=f"Top-1 Acc. ({fixed_label})", zorder=3)
    ax.plot(x, y_top5, marker="s", linewidth=1.8, markersize=6,
            color="#d62728", markeredgecolor="white", markeredgewidth=0.8,
            label=f"Top-5 Acc. ({fixed_label})", zorder=3)

    for xi, yi in zip(x, y_top1):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color="#1f77b4")
    for xi, yi in zip(x, y_top5):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7.5, color="#d62728")

    y_min = min(y_top1.min(), y_top5.min())
    y_max = max(y_top1.max(), y_top5.max())
    margin = (y_max - y_min) * 0.35
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=8, edgecolor="0.6",
              fancybox=False, framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # reset rcParams to default
    matplotlib.rcdefaults()


# =========================================================
# 14. KD GAIN TABLE
# =========================================================
def build_kd_gain_table(results_df):
    rows = []

    students = sorted(set(
        r for r in results_df["model"].tolist()
        if r in STUDENT_MODELS
    ))

    for student in students:
        std_row = results_df[
            (results_df["model"] == student) &
            (results_df["training_type"] == "standard")
        ].iloc[0]

        kd_row = results_df[
            (results_df["model"] == student) &
            (results_df["training_type"] == "kd")
        ].iloc[0]

        rows.append({
            "student": student,
            "standard_top1": float(std_row["top1"]),
            "kd_top1": float(kd_row["top1"]),
            "top1_gain": float(kd_row["top1"] - std_row["top1"]),
            "standard_top5": float(std_row["top5"]),
            "kd_top5": float(kd_row["top5"]),
            "top5_gain": float(kd_row["top5"] - std_row["top5"]),
            "standard_latency_ms": float(std_row["latency_ms"]),
            "kd_latency_ms": float(kd_row["latency_ms"]),
            "latency_diff_ms": float(kd_row["latency_ms"] - std_row["latency_ms"]),
        })

    return pd.DataFrame(rows)


# =========================================================
# 15. MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-only", action="store_true",
                        help="Skip full training, load teacher checkpoint and run ablation only.")
    args = parser.parse_args()

    print("=" * 90)
    print("Enhancing Lightweight CNNs through Knowledge Distillation for Image Classification")
    print("=" * 90)
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, train_len, val_len, test_len = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO
    )

    print(f"Train samples      : {train_len}")
    print(f"Validation samples : {val_len}")
    print(f"Test samples       : {test_len}")

    if not args.ablation_only:
        all_results = []

        # A. Teacher
        teacher_model, teacher_history, teacher_result = train_standard_model(
            model_name=TEACHER_MODEL,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=EPOCHS_TEACHER,
            device=DEVICE
        )
        teacher_result["training_type"] = "teacher"
        all_results.append(teacher_result)

        # B. Students
        for student_name in STUDENT_MODELS:
            # Standard student
            student_std_model, history_std, result_std = train_standard_model(
                model_name=student_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=EPOCHS_STUDENT,
                device=DEVICE
            )
            all_results.append(result_std)

            # KD student
            student_kd_model, history_kd, result_kd = train_kd_student(
                student_name=student_name,
                teacher_model=teacher_model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=EPOCHS_STUDENT,
                device=DEVICE
            )
            all_results.append(result_kd)

            # Plot training curve
            plot_training_curve(
                student_name=student_name,
                history_std=history_std,
                history_kd=history_kd,
                save_path=os.path.join(FIG_DIR, f"curve_{student_name}.png")
            )

        # Overall results
        results_df = pd.DataFrame(all_results)
        results_df = results_df[
            [
                "model", "training_type", "params_m", "best_val_top1",
                "test_loss", "top1", "top5", "latency_ms",
                "train_time_sec", "checkpoint", "history_path"
            ]
        ]

        results_csv = os.path.join(OUTPUT_DIR, "results_summary.csv")
        results_json = os.path.join(OUTPUT_DIR, "results_summary.json")
        results_df.to_csv(results_csv, index=False)
        results_df.to_json(results_json, orient="records", indent=2)

        # KD gain
        gain_df = build_kd_gain_table(results_df)
        gain_csv = os.path.join(OUTPUT_DIR, "kd_gain_summary.csv")
        gain_json = os.path.join(OUTPUT_DIR, "kd_gain_summary.json")
        gain_df.to_csv(gain_csv, index=False)
        gain_df.to_json(gain_json, orient="records", indent=2)

        # Figures
        plot_bar_top1(
            results_df=results_df,
            save_path=os.path.join(FIG_DIR, "bar_top1_comparison.png")
        )

        plot_scatter_accuracy_vs_params(
            results_df=results_df,
            save_path=os.path.join(FIG_DIR, "scatter_accuracy_vs_params.png")
        )

        plot_scatter_accuracy_vs_latency(
            results_df=results_df,
            save_path=os.path.join(FIG_DIR, "scatter_accuracy_vs_latency.png")
        )

        plot_kd_gain(
            gain_df=gain_df,
            save_path=os.path.join(FIG_DIR, "bar_kd_gain.png")
        )

        print("\n=== RESULTS SUMMARY ===")
        print(results_df)

        print("\n=== KD GAIN SUMMARY ===")
        print(gain_df)

        print(f"\nAll outputs saved in: {OUTPUT_DIR}")

    # =========================================================
    # C. ABLATION STUDY (MobileNetV2)
    # =========================================================
    print("\n" + "=" * 90)
    print("ABLATION STUDY: MobileNetV2 Knowledge Distillation Hyperparameters")
    print("=" * 90)

    # Load teacher from checkpoint
    teacher_ckpt_path = os.path.join(CKPT_DIR, f"{TEACHER_MODEL}_standard_best.pt")
    teacher_model = create_model(TEACHER_MODEL, NUM_CLASSES, pretrained=False).to(DEVICE)
    ckpt = torch.load(teacher_ckpt_path, map_location=DEVICE, weights_only=True)
    teacher_model.load_state_dict(ckpt["state_dict"])
    teacher_model.eval()
    print(f"[INFO] Teacher loaded from {teacher_ckpt_path}")

    all_ablation_results = []

    # Experiment 1: Temperature sweep
    temp_results = run_kd_ablation(
        student_name=ABLATION_STUDENT,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        param_type="temperature",
        param_values=ABLATION_TEMPERATURES,
        fixed_alpha=ABLATION_FIXED_ALPHA,
        fixed_temperature=ABLATION_FIXED_TEMPERATURE,
        epochs=ABLATION_EPOCHS,
        device=DEVICE,
    )
    all_ablation_results.extend(temp_results)

    # Experiment 2: Alpha sweep
    alpha_results = run_kd_ablation(
        student_name=ABLATION_STUDENT,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        param_type="alpha",
        param_values=ABLATION_ALPHAS,
        fixed_alpha=ABLATION_FIXED_ALPHA,
        fixed_temperature=ABLATION_FIXED_TEMPERATURE,
        epochs=ABLATION_EPOCHS,
        device=DEVICE,
    )
    all_ablation_results.extend(alpha_results)

    # Save ablation results
    ablation_df = pd.DataFrame(all_ablation_results)
    ablation_df = ablation_df[[
        "experiment_type", "parameter_value", "alpha", "temperature",
        "best_val_top1", "top1", "top5", "latency_ms", "train_time_sec",
    ]]

    ablation_csv = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    ablation_json = os.path.join(OUTPUT_DIR, "ablation_results.json")
    ablation_df.to_csv(ablation_csv, index=False)
    ablation_df.to_json(ablation_json, orient="records", indent=2)

    # Ablation plots
    plot_ablation_curve(
        ablation_df=ablation_df,
        param_type="temperature",
        save_path=os.path.join(FIG_DIR, "ablation_top1_vs_temperature.png"),
    )
    plot_ablation_curve(
        ablation_df=ablation_df,
        param_type="alpha",
        save_path=os.path.join(FIG_DIR, "ablation_top1_vs_alpha.png"),
    )

    print("\n=== ABLATION RESULTS ===")
    print(ablation_df.to_string(index=False))
    print(f"\nAblation outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()