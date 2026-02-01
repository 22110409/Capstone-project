# Client/client.py

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from Client.data_utils import clean_dataframe


class FederatedClient:
    def __init__(
        self,
        client_id,
        data_path,
        feature_cols=None,
        label_col="Label",
        device="cpu",
        balance_strategy="none",  # none | undersample | oversample | class_weight
    ):
        self.client_id = client_id
        self.data_path = data_path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.device = device
        self.balance_strategy = balance_strategy

        self.model = None
        self.data_size = 0

    
    def _load_raw_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df = clean_dataframe(df, self.feature_cols)
        df = df.drop_duplicates()
        df = df.dropna(subset=[self.label_col])
        return df

    def _encode_y(self, y_series: pd.Series) -> np.ndarray:
        # Benign which mean noraml 0, anything else -> 1
        return (
            y_series
            .apply(lambda v: 0 if str(v).strip().upper() == "BENIGN" else 1)
            .astype(int)
            .values
        )

    def _numeric_X(self, X_df: pd.DataFrame, fill_value: float = 0.0) -> np.ndarray:
        X = X_df.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce", downcast="float")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(fill_value, inplace=True)
        return X.values

    def _apply_balance(self, X: np.ndarray, y: np.ndarray):
        """
        Apply balancing ONLY on TRAIN data.
        """
        strategy = (self.balance_strategy or "none").lower()

        if strategy in {"none", "class_weight"}:
            return X, y

        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]

        if len(idx0) == 0 or len(idx1) == 0:
            return X, y

        maj_idx, min_idx = (idx0, idx1) if len(idx0) >= len(idx1) else (idx1, idx0)

        if strategy == "undersample":
            maj_down = resample(
                maj_idx,
                replace=False,
                n_samples=len(min_idx),
                random_state=42,
            )
            new_idx = np.concatenate([maj_down, min_idx])
        else:  
            min_up = resample(
                min_idx,
                replace=True,
                n_samples=len(maj_idx),
                random_state=42,
            )
            new_idx = np.concatenate([maj_idx, min_up])

        np.random.shuffle(new_idx)
        return X[new_idx], y[new_idx]

    def _pos_weight_tensor(self, y_train: np.ndarray) -> torch.Tensor:
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        w = 1.0 if pos <= 0 else neg / pos
        return torch.tensor([w], dtype=torch.float32, device=self.device)


    def get_train_val_test_tensors(
        self,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    ):
        df = self._load_raw_df()

        y = self._encode_y(df[self.label_col])
        X_df = df.drop(columns=[self.label_col])

        stratify = y if len(np.unique(y)) == 2 else None

        X_tmp, X_test_df, y_tmp, y_test = train_test_split(
            X_df,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        val_ratio = val_size / (1.0 - test_size)
        stratify_tmp = y_tmp if len(np.unique(y_tmp)) == 2 else None

        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_tmp,
            y_tmp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify_tmp
        )

        X_train = self._numeric_X(X_train_df)
        X_val = self._numeric_X(X_val_df)
        X_test = self._numeric_X(X_test_df)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_train, y_train = self._apply_balance(X_train, y_train)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(1)

        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device).unsqueeze(1)

        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.data_size = len(y_train)

        return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t


    def get_train_test_tensors(self, test_size=0.2, random_state=42):
        df = self._load_raw_df()

        y = self._encode_y(df[self.label_col])
        X_df = df.drop(columns=[self.label_col])

        stratify = y if len(np.unique(y)) == 2 else None

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        X_train = self._numeric_X(X_train_df)
        X_test = self._numeric_X(X_test_df)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, y_train = self._apply_balance(X_train, y_train)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(1)

        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.data_size = len(y_train)

        return X_train_t, y_train_t, X_test_t, y_test_t

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def get_weights(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}


    def train(self, epochs=10, lr=0.001):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        X_train, y_train, _, _, _, _ = self.get_train_val_test_tensors(
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )

        if (self.balance_strategy or "none").lower() == "class_weight":
            pos_weight = self._pos_weight_tensor(
                y_train.squeeze(1).cpu().numpy().astype(int)
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        return self.get_weights(), self.data_size
