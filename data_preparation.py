import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""
# Pytorch version

# MNIST
# def load_partition(dataset, validation_split, batch_size):
#     """
#     The variables train_loader, val_loader, and test_loader must be returned fixedly.
#     """
#     now = datetime.now()
#     now_str = now.strftime('%Y-%m-%d %H:%M:%S')
#     fl_task = {"dataset": dataset, "start_execution_time": now_str}
#     fl_task_json = json.dumps(fl_task)
#     logging.info(f'FL_Task - {fl_task_json}')

#     # MNIST Data Preprocessing
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
#     ])

#     # Download MNIST Dataset
#     full_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)

#     # Splitting the full dataset into train, validation, and test sets
#     test_split = 0.2
#     train_size = int((1 - validation_split - test_split) * len(full_dataset))
#     validation_size = int(validation_split * len(full_dataset))
#     test_size = len(full_dataset) - train_size - validation_size
#     train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

#     # DataLoader for training, validation, and test
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     return train_loader, val_loader, test_loader
def create_sequences(features: np.ndarray, window_size: int):
    """features: (N, F) → (N-window, window, F), label은 다음 시점의 features[:, 0](TotalIntensity)."""
    seqs, labels = [], []
    for i in range(len(features) - window_size):
        seqs.append(features[i : i + window_size])
        labels.append(features[i + window_size, 0])  # 0번 컬럼이 TotalIntensity
    return np.stack(seqs), np.asarray(labels)

def load_and_prepare(csv_path: str, window_size: int, batch_size):
    EPS = 1e-6
    df = pd.read_csv(csv_path)

    # 안전 파싱
    df["ActivityHour"] = pd.to_datetime(
        df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    df["TotalIntensity"] = pd.to_numeric(df["TotalIntensity"], errors="coerce")
    df["AverageIntensity"] = pd.to_numeric(df["AverageIntensity"], errors="coerce")
    df = df.dropna(subset=["ActivityHour", "TotalIntensity", "AverageIntensity"]).reset_index(drop=True)

    # 가장 데이터가 많은 참가자 선택
    target_id = df["Id"].value_counts().idxmax()
    target = (
        df.loc[df["Id"] == target_id, ["ActivityHour", "TotalIntensity", "AverageIntensity"]]
        .sort_values("ActivityHour")
        .reset_index(drop=True)
        .set_index("ActivityHour")
        .resample("1h")                      # 'H' → 'h' (FutureWarning 방지)
        .mean()
        .ffill()
        .dropna()
        .rename(columns={"TotalIntensity": "TotalIntensity", "AverageIntensity": "AverageIntensity"})
        .reset_index()
    )

    # 시간 주기 특성 (24시간 주기)
    hour = target["ActivityHour"].dt.hour + target["ActivityHour"].dt.minute / 60.0
    target["sin_time"] = np.sin(2 * np.pi * hour / 24)
    target["cos_time"] = np.cos(2 * np.pi * hour / 24)

    # 피처 구성
    feature_cols = ["TotalIntensity", "AverageIntensity", "sin_time", "cos_time"]
    features = target[feature_cols].to_numpy(dtype=np.float32)

    # 시퀀스 생성
    X_all, y_all = create_sequences(features, window_size=window_size)
    total = X_all.shape[0]
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    # 텐서 변환 & 분할
    X_train = torch.tensor(X_all[:train_end], dtype=torch.float32)
    y_train = torch.tensor(y_all[:train_end], dtype=torch.float32)
    X_val   = torch.tensor(X_all[train_end:val_end], dtype=torch.float32)
    y_val   = torch.tensor(y_all[train_end:val_end], dtype=torch.float32)
    X_test  = torch.tensor(X_all[val_end:], dtype=torch.float32)
    y_test  = torch.tensor(y_all[val_end:], dtype=torch.float32)

    # Feature 표준화 (train 기준)
    f_mean = X_train.mean(dim=(0, 1), keepdim=True)
    f_std  = X_train.std(dim=(0, 1), keepdim=True)
    X_train = (X_train - f_mean) / (f_std + EPS)
    X_val   = (X_val   - f_mean) / (f_std + EPS)
    X_test  = (X_test  - f_mean) / (f_std + EPS)

    # Target 표준화 (train 기준)
    y_mean = y_train.mean()
    y_std  = y_train.std()
    y_train_n = (y_train - y_mean) / (y_std + EPS)
    y_val_n   = (y_val   - y_mean) / (y_std + EPS)
    y_test_n  = (y_test  - y_mean) / (y_std + EPS)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val_n),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test_n),  batch_size=batch_size, shuffle=False)

    meta = {
        "feature_mean": f_mean, "feature_std": f_std,
        "target_mean": y_mean,  "target_std": y_std,
        "input_dim": X_train.shape[-1],
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "target_id": int(target_id),
    }
    return train_loader, val_loader, test_loader, meta

def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')


    # iscal setting
    WINDOW_SIZE = 30
    CSV_PATH = "/home/ubuntu/isfolder/fl_agent_paper/buildmodel/content/hourlyIntensities_merged.csv"
    train_loader, val_loader, test_loader, meta = load_and_prepare(CSV_PATH, WINDOW_SIZE, batch_size)


    return train_loader, val_loader, test_loader

# def gl_model_torch_validation(batch_size):
#     """
#     Setting up a dataset to evaluate a global model on the server
#     """
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
#     ])

#     # Load the test set of MNIST Dataset
#     val_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)

#     # DataLoader for validation
#     gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     return gl_val_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """

    EPS = 1e-6
    WINDOW_SIZE = 30
    CSV_PATH = "/app/code/data/hourlyIntensities_merged.csv"

    train_loader, val_loader, test_loader, meta = load_and_prepare(CSV_PATH, WINDOW_SIZE, batch_size)
    
    gl_val_loader = val_loader


    return gl_val_loader