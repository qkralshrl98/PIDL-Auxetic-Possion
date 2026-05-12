import os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

DATA_PATH = "final_combined_data.xlsx"
RESULT_DIR = "resnet_baseline_results"
os.makedirs(RESULT_DIR, exist_ok=True)

df = pd.read_excel(DATA_PATH)
df = df[["t1","t2","d","cell","property","poisson_ratio"]].dropna().copy()
df["cell"] = pd.to_numeric(df["cell"], errors="coerce")

feat_cols = ["t1","t2","d","cell","property"]

train_list, val_list = [], []
for c in [1,3,5]:
    df_c = df[df["cell"]==c].copy()
    y = df_c["poisson_ratio"].values
    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    tr, val = train_test_split(df_c, test_size=0.2, stratify=y_bins, random_state=SEED)
    train_list.append(tr); val_list.append(val)
train_df = pd.concat(train_list).reset_index(drop=True)
val_df   = pd.concat(val_list).reset_index(drop=True)

scaler_X = StandardScaler().fit(train_df[feat_cols].values)
scaler_y = StandardScaler().fit(train_df["poisson_ratio"].values.reshape(-1,1))
def prep(df_):
    X = scaler_X.transform(df_[feat_cols].values).astype(np.float32)
    y = scaler_y.transform(df_["poisson_ratio"].values.reshape(-1,1)).astype(np.float32)
    return X, y
X_train, y_train = prep(train_df)
X_val,   y_val   = prep(val_df)

class ResNetMLP(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.d1 = Dense(256, activation="tanh")
        self.d2 = Dense(256, activation=None)
        self.skip1 = Dense(256, activation=None, use_bias=False)
        self.d3 = Dense(128, activation="tanh")
        self.d4 = Dense(128, activation=None)
        self.skip2 = Dense(128, activation=None, use_bias=False)
        self.d5 = Dense(64, activation="tanh")
        self.out = Dense(1)

    def call(self, x, training=False):
        z1 = self.d1(x)
        z_main = self.d2(z1)
        z_skip = self.skip1(x)
        z = tf.nn.tanh(z_main + z_skip)
        y1 = self.d3(z)
        y_main = self.d4(y1)
        y_skip = self.skip2(z)
        y = tf.nn.tanh(y_main + y_skip)
        y = self.d5(y)
        return self.out(y)

model = ResNetMLP(input_dim=X_train.shape[1])
model.compile(optimizer=Adam(1e-4),
              loss="mse",
              metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                       tf.keras.metrics.MeanSquaredError(name="mse")])

ckpt = ModelCheckpoint(
    filepath=os.path.join(RESULT_DIR, "resnet_baseline_pretrained.h5"),
    monitor="val_mae",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=0
)

hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=100, batch_size=64, verbose=1,
                 callbacks=[ckpt])

model.load_weights(os.path.join(RESULT_DIR, "resnet_baseline_pretrained.h5"))

y_val_pred = model.predict(X_val, verbose=0)
y_val_true_inv = scaler_y.inverse_transform(y_val)
y_val_pred_inv = scaler_y.inverse_transform(y_val_pred)

mse = mean_squared_error(y_val_true_inv, y_val_pred_inv); rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val_true_inv, y_val_pred_inv)
r2  = r2_score(y_val_true_inv, y_val_pred_inv)
rho = spearmanr(y_val_true_inv.ravel(), y_val_pred_inv.ravel())[0]

print(f"\n Best Val Results: "
      f"MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, "
      f"R²={r2:.4f}, Spearman={rho:.3f}")

plt.figure(figsize=(7,5))
plt.plot(hist.history["mae"], label="Train MAE", color="red")
plt.plot(hist.history["val_mae"], label="Val MAE", color="blue")
plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()
plt.tick_params(axis="both", which="both", direction="in")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "learning_curve_MAE.pdf"))
plt.show()

plt.figure(figsize=(7,5))
plt.plot(hist.history["mse"], label="Train MSE", color="red")
plt.plot(hist.history["val_mse"], label="Val MSE", color="blue")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
plt.tick_params(axis="both", which="both", direction="in")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "learning_curve_MSE.pdf"))
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(
    y_val_true_inv, y_val_pred_inv,
    c="#0000FF", s=15, alpha=0.5, edgecolors="none"
)
mn, mx = y_val_true_inv.min(), y_val_true_inv.max()
plt.plot([mn,mx],[mn,mx],"r-",lw=2,label="Ideal")
plt.xlabel("True"); plt.ylabel("Predicted"); plt.legend()
plt.tick_params(axis="both", which="both", direction="in")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "scatter_true_vs_pred.pdf"))
plt.show()
