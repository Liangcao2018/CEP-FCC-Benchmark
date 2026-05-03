"""
FCC Co-processing Comprehensive Benchmark (Final Version)
"""
import pandas as pd
import numpy as np
import logging
import os
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Sklearn Models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
    Lars, LassoLars, BayesianRidge, PassiveAggressiveRegressor,
    SGDRegressor, OrthogonalMatchingPursuit, ARDRegression,
    TheilSenRegressor, RANSACRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except (ImportError, OSError, Exception) as e:
    logging.warning(f"LightGBM failed to load (environment compatibility issue): {e}")
    HAS_LGBM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# SOTA Model Definitions
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_fc = nn.Linear(d_model, 1)
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_fc(x[:, -1, :])

class TCNModel(nn.Module):
    def __init__(self, input_dim, num_channels=[64, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers += [nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0),
                      nn.Conv1d(in_ch, num_channels[i], kernel_size, dilation=dilation),
                      nn.ReLU(), nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        return self.fc(y[:, :, -1])

class MambaLite(nn.Module):
    """A simplified Mamba-like recurrent structure that simulates state transitions via linear scan"""
    def __init__(self, input_dim, d_state=16, d_model=64):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.dt_proj = nn.Linear(d_model, 1)
        self.A = nn.Parameter(torch.randn(1, d_model, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(1, d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(1, d_state, 1) * 0.1)
        self.out_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape
        x = self.input_fc(x) # (batch, seq_len, d_model)

        # Simplified linear scan: simulates RNN/SSM sequence processing
        # Simplified here as global weighted pooling, but in real Mamba it is recursive accumulation
        dt = torch.sigmoid(self.dt_proj(x)) # (batch, seq_len, 1)
        # Simulated state evolution: simple weighted aggregation
        context = torch.sum(x * dt, dim=1) / (torch.sum(dt, dim=1) + 1e-6)
        return self.out_fc(context)

class SeqWrapper(nn.Module):
    def __init__(self, rnn_module, hidden_dim):
        super().__init__()
        self.rnn = rnn_module
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# ==========================================
# Benchmark Main Class
# ==========================================

class ComprehensiveFCCBenchmark:
    def __init__(self, data_path, window_size=10):
        self.data_path = data_path
        self.window_size = window_size
        self.results = []
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_sequences(self, X, y):
        """Create sliding-window sequences"""
        Xs, ys = [], []
        for i in range(len(X) - self.window_size):
            Xs.append(X[i:(i + self.window_size)])
            ys.append(y[i + self.window_size])
        return np.array(Xs), np.array(ys)

    def load_and_preprocess(self):
        logging.info("="*70)
        logging.info(f"Loading and preprocessing data (Window Size: {self.window_size})...")
        df = pd.read_csv(self.data_path)
        if df.columns[0] in ['Unnamed: 0', '']:
            df = df.iloc[:, 1:]

        # Anonymization
        df.columns = ['Target'] + [f'Feature_{i:02d}' for i in range(1, len(df.columns))]
        df = df.interpolate(method='linear').ffill().bfill()

        X_raw = df.drop(columns=['Target']).values
        y_raw = df[['Target']].values

        X_scaled = self.scaler_x.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)

        n = len(X_seq)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        self.X_train = X_seq[:train_end]
        self.X_val = X_seq[train_end:val_end]
        self.X_test = X_seq[val_end:]
        self.y_train = y_seq[:train_end]
        self.y_val = y_seq[train_end:val_end]
        self.y_test = y_seq[val_end:]

        logging.info(f"Total sequences generated: {n}")
        logging.info(f"Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")
        logging.info(f"Input shape: {self.X_train.shape} | Target shape: {self.y_train.shape}")
        logging.info("="*70)

    def calculate_metrics(self, y_true_sc, y_pred_sc, y_train_true_sc, y_train_pred_sc, name):
        """Compute metrics on both training and test sets (normalized)"""
        # Test set
        test_rmse = np.sqrt(mean_squared_error(y_true_sc, y_pred_sc))
        test_mae = mean_absolute_error(y_true_sc, y_pred_sc)
        test_r2 = r2_score(y_true_sc, y_pred_sc)

        # Training set
        train_rmse = np.sqrt(mean_squared_error(y_train_true_sc, y_train_pred_sc))
        train_mae = mean_absolute_error(y_train_true_sc, y_train_pred_sc)
        train_r2 = r2_score(y_train_true_sc, y_train_pred_sc)

        # MAPE (computed on inverse-normalized values)
        y_test_orig = self.scaler_y.inverse_transform(y_true_sc.reshape(-1, 1)).flatten()
        y_pred_orig = self.scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()

        # Avoid division by zero
        mask = y_test_orig != 0
        mape = np.mean(np.abs((y_test_orig[mask] - y_pred_orig[mask]) / (y_test_orig[mask]))) * 100

        result = {
            'Model': name,
            'Train_N-RMSE': train_rmse, 'Train_N-MAE': train_mae, 'Train_R2': train_r2,
            'Test_N-RMSE': test_rmse, 'Test_N-MAE': test_mae, 'Test_R2': test_r2,
            'Test_MAPE': mape
        }
        self.results.append(result)
        logging.info(f"{name:20s} | Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
        return result

    def get_all_sklearn_models(self):
        """Return all sklearn models (expanded + SOTA)"""
        return [
            # Linear (14)
            ('OLS', LinearRegression()),
            ('Ridge', Ridge()),
            ('Lasso', Lasso(alpha=0.01)),
            ('ElasticNet', ElasticNet(alpha=0.01)),
            ('Huber', HuberRegressor()),
            ('Lars', Lars()),
            ('LassoLars', LassoLars(alpha=0.01)),
            ('BayesianRidge', BayesianRidge()),
            ('PassiveAggressive', PassiveAggressiveRegressor(max_iter=1000)),
            ('SGDRegressor', SGDRegressor(max_iter=1000, tol=1e-3)),
            ('OMP', OrthogonalMatchingPursuit()),
            ('ARDRegression', ARDRegression()),
            ('TheilSen', TheilSenRegressor()),
            ('RANSAC', RANSACRegressor()),

            # PLS
            ('PLS', PLSRegression(n_components=5)),

            # Ensemble (8)
            ('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1)),
            ('ExtraTrees', ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1)),
            ('AdaBoost', AdaBoostRegressor(n_estimators=100)),
            ('GradientBoosting', GradientBoostingRegressor(n_estimators=100)),
            ('HistGradientBoosting', HistGradientBoostingRegressor(max_iter=100)),
            ('Bagging', BaggingRegressor(n_estimators=20, n_jobs=-1)),
            ('XGBoost', XGBRegressor(n_estimators=100, n_jobs=-1)),
        ]
        if HAS_LGBM:
            models.append(('LGBM', LGBMRegressor(n_estimators=100, verbose=-1)))

        models += [
            # Neighbors & Others (6)
            ('KNN', KNeighborsRegressor(n_neighbors=5)),
            ('DecisionTree', DecisionTreeRegressor(max_depth=15)),
            ('SVR-Linear', LinearSVR(max_iter=2000)),
            ('SVR-RBF', SVR(kernel='rbf', C=1.0, cache_size=700)),
            ('MLP-Sklearn', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)),
            ('KernelRidge', KernelRidge(alpha=1.0))
        ]

    def run_sklearn_benchmarks(self):
        models = self.get_all_sklearn_models()
        logging.info(f"\nStarting {len(models)} Sklearn/traditional models (using flattened sequences)...")

        # Flatten 3D sequences into a 2D matrix for sklearn compatibility
        X_train_flat = self.X_train.reshape(len(self.X_train), -1)
        X_test_flat = self.X_test.reshape(len(self.X_test), -1)

        for name, model in models:
            try:
                start = time.time()
                using_full_data = True

                # Downsample for compute-intensive models
                if name in ['TheilSen', 'SVR-RBF', 'KernelRidge'] and len(X_train_flat) > 15000:
                    logging.info(f"  {name} has high compute complexity; downsampling to 15000 rows for training...")
                    idx = np.random.choice(len(X_train_flat), 15000, replace=False)
                    model.fit(X_train_flat[idx], self.y_train[idx].ravel())
                    using_full_data = False
                    y_train_pred = model.predict(X_train_flat[idx])
                    y_train_true = self.y_train[idx].ravel()
                else:
                    model.fit(X_train_flat, self.y_train.ravel())
                    y_train_pred = model.predict(X_train_flat)
                    y_train_true = self.y_train.ravel()

                y_test_pred = model.predict(X_test_flat)
                self.calculate_metrics(self.y_test.ravel(), y_test_pred, y_train_true, y_train_pred, name)

                data_status = "Full Data" if using_full_data else "Sampled"
                logging.info(f"  [{data_status}] Elapsed: {time.time()-start:.1f}s")
            except Exception as e:
                logging.error(f"{name} failed: {e}")

    def run_pytorch_benchmarks(self, epochs=50):
        """Run all PyTorch deep-learning models with time-series sequence support"""
        # X shape: (batch, seq_len, features)
        in_dim = self.X_train.shape[2]
        hid = 64  # Unified hidden dimension

        models = [
            # DNN variants (flatten input for DNN)
            ("DNN-Small", nn.Sequential(nn.Flatten(), nn.Linear(in_dim * self.window_size, hid), nn.ReLU(), nn.Linear(hid, 1))),
            ("DNN-Medium", nn.Sequential(nn.Flatten(), nn.Linear(in_dim * self.window_size, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))),
            ("DNN-Large", nn.Sequential(nn.Flatten(), nn.Linear(in_dim * self.window_size, hid*2), nn.ReLU(), nn.Linear(hid*2, hid), nn.ReLU(), nn.Linear(hid, 1))),

            # Sequential models (native 3D sequence handling)
            ("Transformer", TransformerRegressor(in_dim, d_model=hid)),
            ("TCN", TCNModel(in_dim, num_channels=[hid, hid])),
            ("Mamba-Lite", MambaLite(in_dim, d_model=hid)),
            ("LSTM", SeqWrapper(nn.LSTM(in_dim, hid, batch_first=True), hid)),
            ("GRU", SeqWrapper(nn.GRU(in_dim, hid, batch_first=True), hid))
        ]

        logging.info(f"\nStarting {len(models)} deep-learning models (Epochs={epochs})...")
        for name, model in models:
            self._train_pytorch(name, model, epochs)

    def _train_pytorch(self, name, model, epochs):
        try:
            start = time.time()
            model = model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            # Increase batch_size for faster training
            loader = DataLoader(TensorDataset(torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train)),
                              batch_size=512, shuffle=True)

            for ep in range(epochs):
                model.train()
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                y_train_pred = model(torch.FloatTensor(self.X_train).to(self.device)).cpu().numpy()
                y_test_pred = model(torch.FloatTensor(self.X_test).to(self.device)).cpu().numpy()

            self.calculate_metrics(self.y_test.ravel(), y_test_pred.ravel(),
                                 self.y_train.ravel(), y_train_pred.ravel(), name)

            # Save predictions of the strong sequential model (Transformer) for plotting
            if name == "Transformer":
                # Save test set
                test_pred_df = pd.DataFrame({
                    'Actual': self.y_test.ravel(),
                    'Predicted': y_test_pred.ravel()
                })
                test_pred_df.to_csv('best_predictions_test.csv', index=False)

                # Save training set
                train_pred_df = pd.DataFrame({
                    'Actual': self.y_train.ravel(),
                    'Predicted': y_train_pred.ravel()
                })
                train_pred_df.to_csv('best_predictions_train.csv', index=False)
                logging.info(f"  Exported {name} (Time-Series) train/test predictions for plotting")

            logging.info(f"  Elapsed: {time.time()-start:.1f}s")
        except Exception as e:
            logging.error(f"{name} failed: {e}")

    def report(self):
        df = pd.DataFrame(self.results).sort_values(by='Test_R2', ascending=False)

        print("\n" + "="*100)
        print("FCC Co-processing Comprehensive Benchmark Final Report (ALL MODELS)")
        print("="*100)
        print(f"Total models tested: {len(df)}")
        print(f"Dataset size: Train={len(self.X_train)}, Test={len(self.X_test)}")
        print("="*100)
        # Formatted output, 4 decimal places
        print(df.round(4).to_string(index=False))
        print("="*100)

        df.to_csv('benchmark_comprehensive_final.csv', index=False)
        logging.info("\nFinal results saved to: benchmark_comprehensive_final.csv")

if __name__ == "__main__":
    DATA_PATH = '/content/drive/MyDrive/green LCC new final smoothed.csv'
    if os.path.exists(DATA_PATH):
        bm = ComprehensiveFCCBenchmark(DATA_PATH)
        bm.load_and_preprocess()
        bm.run_sklearn_benchmarks()      # 30+ models
        bm.run_pytorch_benchmarks(epochs=50)  # 8 DL models
        bm.report()
    else:
        logging.error(f"Data file not found: {DATA_PATH}")
