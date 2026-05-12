# sweep_train_resnet_tanh.py
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
RESULT_DIR = "alpha_sweep_results_resnet_tanh"
os.makedirs(RESULT_DIR, exist_ok=True)

MARGIN = 0.0
ALPHA_LIST = [0.01, 0.1, 1, 10, 100]

df = pd.read_excel(DATA_PATH)
df = df[["t1","t2","d","cell","property","poisson_ratio"]].dropna().copy()
df["cell"] = pd.to_numeric(df["cell"], errors="coerce")

feat_cols = ["t1","t2","d","cell","property"]
cell_idx  = feat_cols.index("cell")

train_list, val_list = [], []
for c in [1,3]:
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

class MonotonicResNetTanh(Model):
    def __init__(self, input_dim, cell_col_idx, margin=0.0, alpha=1.0):
        super().__init__()
        self.cell_col_idx = int(cell_col_idx)
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.d1 = Dense(256, activation="tanh")
        self.d2 = Dense(256, activation=None)
        self.skip1 = Dense(256, activation=None, use_bias=False)
        self.d3 = Dense(128, activation="tanh")
        self.d4 = Dense(128, activation=None)
        self.skip2 = Dense(128, activation=None, use_bias=False)
        self.d5 = Dense(64, activation="tanh")
        self.out = Dense(1)

        self.mse_fn = tf.keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        z = self.d1(x)
        z_main = self.d2(z)
        z_skip = self.skip1(x)
        z = tf.nn.tanh(z_main + z_skip)
        y = self.d3(z)
        y_main = self.d4(y)
        y_skip = self.skip2(z)
        y = tf.nn.tanh(y_main + y_skip)
        y = self.d5(y)
        return self.out(y)

    def _mono_violation_loss(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            y_ = self.call(x, training=True)
        J = g.batch_jacobian(y_, x)
        grads = tf.squeeze(J, axis=1)
        dY_dCell = grads[:, self.cell_col_idx]
        violation = tf.nn.relu(self.margin - dY_dCell)
        return tf.reduce_mean(tf.square(violation))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.call(x, training=True)
            data_loss = self.mse_fn(y, y_pred)
            mono_loss = self._mono_violation_loss(x)
            loss = data_loss + self.alpha * mono_loss
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "mse": data_loss, "mono_loss": mono_loss})
        return results

alpha_results = []
for alpha in ALPHA_LIST:
    print(f"\n=== Training with alpha={alpha} ===")
    model_tmp = MonotonicResNetTanh(input_dim=X_train.shape[1], cell_col_idx=cell_idx,
                                    margin=MARGIN, alpha=alpha)
    model_tmp.compile(optimizer=Adam(1e-4),
                      loss="mse",
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                               tf.keras.metrics.MeanSquaredError(name="mse")])

    ckpt_tmp = ModelCheckpoint(
        filepath=os.path.join(RESULT_DIR, f"tmp_best_alpha{alpha}.weights.h5"),
        monitor="val_mae",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=0
    )

    hist = model_tmp.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=100, batch_size=64, verbose=0,
                         callbacks=[ckpt_tmp])

    model_tmp.load_weights(os.path.join(RESULT_DIR, f"tmp_best_alpha{alpha}.weights.h5"))
    yv_pred = model_tmp.predict(X_val, verbose=0)
    yv_true = scaler_y.inverse_transform(y_val); yv_pred_inv = scaler_y.inverse_transform(yv_pred)

    mse = mean_squared_error(yv_true, yv_pred_inv); rmse = np.sqrt(mse)
    mae = mean_absolute_error(yv_true, yv_pred_inv)
    r2  = r2_score(yv_true, yv_pred_inv)
    rho = spearmanr(yv_true.ravel(), yv_pred_inv.ravel())[0]

    print(f"  -> [Best Epoch] Val MSE={mse:.6f}, RMSE={rmse:.6f}, "
          f"MAE={mae:.6f}, R²={r2:.4f}, Spearman={rho:.3f}")

    alpha_results.append({"alpha":alpha,"MSE":mse,"RMSE":rmse,"MAE":mae,
                          "R2":r2,"Spearman":rho})

df_alpha = pd.DataFrame(alpha_results)
df_alpha.to_excel(os.path.join(RESULT_DIR,"alpha_sweep_results.xlsx"), index=False)

best_row = df_alpha.loc[df_alpha["MAE"].idxmin()]
best_alpha = float(best_row["alpha"])
print(f"\n Best alpha selected: {best_alpha}")

model_best = MonotonicResNetTanh(input_dim=X_train.shape[1], cell_col_idx=cell_idx,
                                 margin=MARGIN, alpha=best_alpha)
model_best.compile(optimizer=Adam(1e-3),
                   loss="mse",
                   metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                            tf.keras.metrics.MeanSquaredError(name="mse")])

ckpt = ModelCheckpoint(
    filepath=os.path.join(RESULT_DIR, "pretrained_best.weights.h5"),
    monitor="val_mae",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=0
)

hist_best = model_best.fit(X_train, y_train, validation_data=(X_val, y_val),
                           epochs=100, batch_size=64, verbose=0,
                           callbacks=[ckpt])

plt.figure(figsize=(7,5))
plt.plot(hist_best.history["mae"], label="Train MAE", color="red")
plt.plot(hist_best.history["val_mae"], label="Val MAE", color="blue")
plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"best_learning_curve_MAE_alpha{best_alpha}.pdf"))
plt.show()

plt.figure(figsize=(7,5))
plt.plot(hist_best.history["mse"], label="Train MSE", color="red")
plt.plot(hist_best.history["val_mse"], label="Val MSE", color="blue")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"best_learning_curve_MSE_alpha{best_alpha}.pdf"))
plt.show()

y_val_pred_best = model_best.predict(X_val, verbose=0)
y_val_true_inv = scaler_y.inverse_transform(y_val)
y_val_pred_inv = scaler_y.inverse_transform(y_val_pred_best)

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
plt.savefig(os.path.join(RESULT_DIR, f"scatter_true_vs_pred_alpha{best_alpha}.pdf"))
plt.show()

metrics = {"MAE":"MAE","RMSE":"RMSE","R2":"R²"}
for metric, title in metrics.items():
    plt.figure(figsize=(7,5))
    plt.plot(df_alpha["alpha"], df_alpha[metric], marker="o", linewidth=2,
             color="blue", markerfacecolor="blue", markeredgecolor="blue")
    plt.xscale("log")
    plt.xlabel("Alpha (PI loss weight, log scale)")
    plt.ylabel(title)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, f"alpha_vs_{metric.replace('/', '_')}_logx.pdf")
    plt.savefig(out_path)
    plt.show()

with open(os.path.join(RESULT_DIR,"best_alpha.txt"), "w") as f:
    f.write(str(best_alpha))
print(f" Saved best model (by val_mae) and best_alpha.txt")
