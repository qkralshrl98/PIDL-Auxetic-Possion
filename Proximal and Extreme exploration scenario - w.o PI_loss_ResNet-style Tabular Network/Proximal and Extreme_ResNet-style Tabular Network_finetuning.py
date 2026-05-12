import os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.ticker as ticker

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 25,
    "axes.labelsize": 25,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 600,  
    "savefig.dpi": 600 
})

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC_OPS"] = "1"
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

DATA_PATH  = "final_combined_data.xlsx"
RESULT_DIR = "resnet_baseline_results"     
OUT_DIR    = "finetune_resnet_results" 
os.makedirs(OUT_DIR, exist_ok=True)

SUPPORT_N = 5
FT_EPOCHS = 50
FT_BATCH  = 2
FT_LR     = 1e-5
MARGIN    = 0.0

df = pd.read_excel(DATA_PATH)
df = df[["t1","t2","d","cell","property","poisson_ratio"]].dropna().copy()
df["cell"] = pd.to_numeric(df["cell"], errors="coerce")

feat_cols = ["t1","t2","d","cell","property"]
cell_idx  = feat_cols.index("cell")

df_source = df[df["cell"].isin([1,3,5])].copy()
scaler_X = StandardScaler().fit(df_source[feat_cols].values)
scaler_y = StandardScaler().fit(df_source["poisson_ratio"].values.reshape(-1,1))
def prep(df_):
    X = scaler_X.transform(df_[feat_cols].values).astype(np.float32)
    y = scaler_y.transform(df_["poisson_ratio"].values.reshape(-1,1)).astype(np.float32)
    return X, y

class MonotonicResNetTanh(Model):
    def __init__(self, input_dim, cell_col_idx, margin=0.0):
        super().__init__()
        self.cell_col_idx = int(cell_col_idx)
        self.margin = float(margin)
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
        z = self.d1(x); z_main = self.d2(z); z_skip = self.skip1(x)
        z = tf.nn.tanh(z_main + z_skip)
        y = self.d3(z); y_main = self.d4(y); y_skip = self.skip2(z)
        y = tf.nn.tanh(y_main + y_skip)
        y = self.d5(y)
        return self.out(y)
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.call(x, training=True)
            loss = self.mse_fn(y, y_pred)  
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

class RootMeanSquaredError(tf.keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs): 
        super().__init__(name=name, **kwargs); 
        self.mse = tf.keras.metrics.MeanSquaredError()
    def update_state(self, y_true, y_pred, sample_weight=None): 
        self.mse.update_state(y_true, y_pred, sample_weight)
    def result(self): 
        return tf.sqrt(self.mse.result())
    def reset_state(self): 
        self.mse.reset_state()

class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name="r2", **kwargs): 
        super().__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name="ssr", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype); y_pred = tf.cast(y_pred, self.dtype)
        ssr = tf.reduce_sum(tf.square(y_true - y_pred)); mean_true = tf.reduce_mean(y_true)
        sst = tf.reduce_sum(tf.square(y_true - mean_true))
        self.ssr.assign_add(ssr); self.sst.assign_add(sst)
    def result(self): 
        return 1.0 - (self.ssr / (self.sst + tf.keras.backend.epsilon()))
    def reset_state(self): 
        self.ssr.assign(0.0); self.sst.assign(0.0)

class QueryMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_query, y_query): 
        super().__init__(); self.Xq, self.yq = X_query, y_query
    def on_epoch_end(self, epoch, logs=None):
        t0 = time.time(); y_pred = self.model.predict(self.Xq, verbose=0)
        elapsed = (time.time() - t0) / len(self.Xq)
        mae = mean_absolute_error(self.yq, y_pred); rmse = np.sqrt(mean_squared_error(self.yq, y_pred)); r2 = r2_score(self.yq, y_pred)
        logs["query_mae"] = mae; logs["query_rmse"] = rmse; logs["query_r2"] = r2; logs["query_infer_time"] = elapsed
        print(f"[Epoch {epoch+1}] Query MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Inference/sample={elapsed:.2e}s")

weights_path = os.path.join(RESULT_DIR, "resnet_baseline_pretrained.h5")

for c in [5,7]:
    print(f"\n================ Fine-tuning on Cell {c} =================")
    df_c = df[df["cell"] == c].copy()
    support_df = df_c.sample(n=SUPPORT_N, random_state=SEED)
    query_df   = df_c.drop(support_df.index)
    X_support, y_support = prep(support_df); X_query, y_query = prep(query_df)

    model = MonotonicResNetTanh(input_dim=len(feat_cols), cell_col_idx=cell_idx, margin=MARGIN)
    _ = model(tf.zeros((1, len(feat_cols)), dtype=tf.float32))
    model.load_weights(weights_path)

    model.compile(
        optimizer=Adam(learning_rate=FT_LR),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            RootMeanSquaredError(name="rmse"),
            R2Score(name="r2"),
        ],
    )

    ckpt_path = os.path.join(OUT_DIR, f"ckpt_cell{c}_best.weights.h5")
    ckpt_cb = ModelCheckpoint(
        ckpt_path, monitor="val_mae", mode="min",
        save_best_only=True, save_weights_only=True, verbose=0
    )

    history = model.fit(
        X_support, y_support,
        validation_data=(X_query, y_query),
        epochs=FT_EPOCHS, batch_size=FT_BATCH, verbose=1,
        callbacks=[QueryMetricsCallback(X_query, y_query), ckpt_cb]
    )

    best_epoch = int(np.argmin(history.history["val_mae"]) + 1)
    print(f" Cell {c}: Best epoch by val_mae: {best_epoch}")

    epochs_range = range(1, FT_EPOCHS + 1)
    plt.figure(figsize=(7,5))
    plt.plot(epochs_range, history.history["mae"], label="Support MAE", color="red")
    plt.plot(epochs_range, history.history["val_mae"], label="Query MAE", color="blue")
    plt.axvline(best_epoch, color="black", linestyle="--", label=f"Optimal Epoch={best_epoch}")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()
    plt.tick_params(direction="in"); plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"ft_curve_MAE_cell{c}_support{SUPPORT_N}_epoch{best_epoch}.pdf")); plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(epochs_range, history.history["mse"], label="Support MSE", color="red")
    plt.plot(epochs_range, history.history["val_mse"], label="Query MSE", color="blue")
    plt.axvline(best_epoch, color="black", linestyle="--", label=f"Optimal Epoch={best_epoch}")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
    plt.tick_params(direction="in"); plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"ft_curve_MSE_cell{c}_support{SUPPORT_N}_epoch{best_epoch}.pdf")); plt.show()

    model_best = MonotonicResNetTanh(input_dim=len(feat_cols), cell_col_idx=cell_idx, margin=MARGIN)
    _ = model_best(tf.zeros((1, len(feat_cols)), dtype=tf.float32))
    model_best.load_weights(ckpt_path)

    y_sup_pred = scaler_y.inverse_transform(model_best.predict(X_support, verbose=0)); y_sup_true = scaler_y.inverse_transform(y_support)
    print(f"\n Cell {c} Support Set (Best Epoch)")
    print("MAE:", mean_absolute_error(y_sup_true, y_sup_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_sup_true, y_sup_pred)))
    print("R²:", r2_score(y_sup_true, y_sup_pred))

    t0 = time.time(); y_q_pred = scaler_y.inverse_transform(model_best.predict(X_query, verbose=0)); infer_per = (time.time() - t0) / len(X_query)
    y_q_true = scaler_y.inverse_transform(y_query)
    print(f"\n Cell {c} Query Set (Best Epoch)")
    print("MAE:", mean_absolute_error(y_q_true, y_q_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_q_true, y_q_pred)))
    print("R²:", r2_score(y_q_true, y_q_pred))
    print("Inference/sample:", infer_per, "sec")

    plt.figure(figsize=(6,6))
    plt.scatter(y_q_true, y_q_pred, c="#0000FF", s=12, alpha=0.35, edgecolors="none")
    mn, mx = y_q_true.min(), y_q_true.max()
    plt.plot([mn,mx],[mn,mx],"r-",lw=2)
    plt.xlabel("True Poisson's Ratio"); plt.ylabel("Predicted Poisson's Ratio")
    plt.tick_params(axis="both", which="both", direction="in")
    
    
    r2_val = r2_score(y_q_true, y_q_pred)
    r2_text = f"R² = {r2_val:.4f}"
    plt.text(0.05, 0.95, r2_text, transform=plt.gca().transAxes,
             fontsize= 30, va="top", ha="left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUT_DIR,
        f"scatter_query_bestEpoch{best_epoch}_support{SUPPORT_N}.tiff"
    ))
    plt.show()


    results_dict = {
        "MAE": [mean_absolute_error(y_q_true, y_q_pred)],
        "RMSE": [np.sqrt(mean_squared_error(y_q_true, y_q_pred))],
        "R2": [r2_val],
        "Inference/sample (s)": [infer_per],
        "Best Epoch": [best_epoch],
        "Support N": [SUPPORT_N],
    }
    df_results = pd.DataFrame(results_dict)
    out_xlsx = os.path.join(OUT_DIR, f"query_best_epoch_results_cell{c}_support{SUPPORT_N}.xlsx")
    df_results.to_excel(out_xlsx, index=False)
    print(f" Saved Query Set best epoch results for Cell {c} to {out_xlsx}")

    extra_results = {}
    
    mask_02 = y_q_true.ravel() < 0.2
    if mask_02.sum() > 0:
        extra_results["R2_true<0.2"] = r2_score(y_q_true[mask_02], y_q_pred[mask_02])
    
    mask_01 = y_q_true.ravel() < 0.1
    if mask_01.sum() > 0:
        extra_results["R2_true<0.1"] = r2_score(y_q_true[mask_01], y_q_pred[mask_01])

    mask_005 = y_q_true.ravel() < 0.05
    if mask_005.sum() > 0:
        extra_results["R2_true<0.05"] = r2_score(y_q_true[mask_005], y_q_pred[mask_005])
    
    sorted_idx = np.argsort(y_q_true.ravel())
    n = len(sorted_idx)
    
    top20 = sorted_idx[int(n*0.8):]
    top10 = sorted_idx[int(n*0.9):]
    top5  = sorted_idx[int(n*0.95):]
    
    extra_results["R2_top20%"] = r2_score(y_q_true[top20], y_q_pred[top20])
    extra_results["R2_top10%"] = r2_score(y_q_true[top10], y_q_pred[top10])
    extra_results["R2_top5%"]  = r2_score(y_q_true[top5],  y_q_pred[top5])
    
    print("\n Additional R² Results (Query Set, Best Epoch):")
    for k,v in extra_results.items():
        print(f"{k}: {v:.4f}")
    
    df_extra = pd.DataFrame([extra_results])
    df_extra.index = ["R2"]
    extra_xlsx = os.path.join(OUT_DIR, f"query_additional_R2_cell{c}_support{SUPPORT_N}.xlsx")
    df_extra.to_excel(extra_xlsx)
    print(f" Saved additional R² results for Cell {c} to {extra_xlsx}")
