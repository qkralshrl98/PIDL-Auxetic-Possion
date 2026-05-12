import os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.ticker as ticker

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, Embedding
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
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

DATA_PATH  = "final_combined_data.xlsx"
PRETRAINED_DIR = "alpha_sweep_results" 
OUT_DIR    = "finetune_results_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

SUPPORT_N = 5
FT_EPOCHS = 20
FT_BATCH  = 2
FT_LR     = 1e-5

df = pd.read_excel(DATA_PATH)
df = df[["t1","t2","d","cell","property","poisson_ratio"]].dropna().copy()
df["cell"] = pd.to_numeric(df["cell"], errors="coerce").astype(int)

num_cols = ["t1","t2","d","property","cell"]  
cat_cols = ["cell"]                         
feat_cols = ["t1","t2","d","cell","property"]

all_cells = sorted(df["cell"].unique().tolist())
cell_to_idx = {c:i for i,c in enumerate(all_cells)}
cat_card = [len(all_cells)]

df_source = df[df["cell"].isin([1,3])].copy()
scaler_X_num = StandardScaler().fit(df_source[num_cols].values)
scaler_y = StandardScaler().fit(df_source["poisson_ratio"].values.reshape(-1,1))

def prep(df_):
    X_num = scaler_X_num.transform(df_[num_cols].values).astype(np.float32)
    X_cat = df_["cell"].map(cell_to_idx).values.astype(np.int32).reshape(-1,1)
    y = scaler_y.transform(df_["poisson_ratio"].values.reshape(-1,1)).astype(np.float32)
    return X_num, X_cat, y

class FTTransformer(Model):
    def __init__(self, num_num, num_cat, categories_cardinalities,
                 d_model=256, num_heads=4, num_layers=3, ff_dim=128, dropout=0.1):
        super().__init__()
        self.num_num = num_num
        self.num_cat = num_cat
        self.d_model = d_model
        self.num_linears = [Dense(d_model, use_bias=True, name=f"num_tok_{j}") for j in range(num_num)]
        self.cat_embs, self.cat_bias = [], []
        for j, card in enumerate(categories_cardinalities):
            self.cat_embs.append(Embedding(input_dim=card, output_dim=d_model, name=f"cat_emb_{j}"))
            self.cat_bias.append(self.add_weight(name=f"cat_bias_{j}", shape=(d_model,),
                                                 initializer="zeros", trainable=True))
        self.cls_token = self.add_weight(name="cls_token", shape=(1, d_model),
                                         initializer="zeros", trainable=True)

        self.enc_layers = []
        for li in range(num_layers):
            self.enc_layers.append([
                LayerNormalization(epsilon=1e-6, name=f"enc{li}_ln1"),
                MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name=f"enc{li}_mha"),
                Dropout(dropout, name=f"enc{li}_drop1"),
                LayerNormalization(epsilon=1e-6, name=f"enc{li}_ln2"),
                Dense(ff_dim, activation="relu", name=f"enc{li}_ff1"),
                Dense(d_model, name=f"enc{li}_ff2"),
                Dropout(dropout, name=f"enc{li}_drop2")
            ])

        self.pred_norm = LayerNormalization(epsilon=1e-6, name="pred_ln")
        self.pred_dense = Dense(d_model, activation="relu", name="pred_dense")
        self.out = Dense(1, name="out")

    def _forward_tokens(self, x_num, x_cat, training=False):
        B = tf.shape(x_num)[0]
        tok_num = []
        for j in range(self.num_num):
            col = tf.expand_dims(x_num[:, j], axis=1) 
            tok_num.append(self.num_linears[j](col))   
        tok_cat = []
        for j in range(self.num_cat):
            e = self.cat_embs[j](x_cat[:, j]) + self.cat_bias[j]
            tok_cat.append(e)
        tokens = tf.stack(tok_num + tok_cat, axis=1)   
        cls_tok = tf.broadcast_to(self.cls_token[None, :, :], [B, 1, self.d_model])
        z = tf.concat([cls_tok, tokens], axis=1)   
        for norm1, attn, drop1, norm2, ffd1, ffd2, drop2 in self.enc_layers:
            z_norm = norm1(z)
            attn_out = attn(z_norm, z_norm)
            z = z + drop1(attn_out, training=training)
            z_norm2 = norm2(z)
            ff_out = ffd2(ffd1(z_norm2))
            z = z + drop2(ff_out, training=training)

        cls_repr = z[:, 0, :]
        h = self.pred_dense(self.pred_norm(cls_repr))
        return self.out(h)

    def call(self, inputs, training=False):
        x_num, x_cat = inputs
        return self._forward_tokens(x_num, x_cat, training=training)

class RootMeanSquaredError(tf.keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
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
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        self.ssr.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        mean_true = tf.reduce_mean(y_true)
        self.sst.assign_add(tf.reduce_sum(tf.square(y_true - mean_true)))
    def result(self):
        return 1.0 - (self.ssr / (self.sst + tf.keras.backend.epsilon()))
    def reset_state(self):
        self.ssr.assign(0.0); self.sst.assign(0.0)

class QueryMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, Xq_num, Xq_cat, yq):
        super().__init__()
        self.Xq_num, self.Xq_cat, self.yq = Xq_num, Xq_cat, yq
    def on_epoch_end(self, epoch, logs=None):
        t0 = time.time()
        y_pred = self.model.predict((self.Xq_num, self.Xq_cat), verbose=0)
        elapsed = (time.time() - t0) / len(self.Xq_num)
        mae = mean_absolute_error(self.yq, y_pred)
        rmse = np.sqrt(mean_squared_error(self.yq, y_pred))
        r2 = r2_score(self.yq, y_pred)
        logs["query_mae"] = mae; logs["query_rmse"] = rmse
        logs["query_r2"] = r2; logs["query_infer_time"] = elapsed
        print(f"[Epoch {epoch+1}] Query MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Inference/sample={elapsed:.2e}s")

weights_path = os.path.join(PRETRAINED_DIR, "pretrained_best.weights.h5")
model = FTTransformer(num_num=len(num_cols), num_cat=len(cat_cols), categories_cardinalities=cat_card)
_ = model((tf.zeros((1, len(num_cols)), dtype=tf.float32),
           tf.zeros((1, len(cat_cols)), dtype=tf.int32)))
model.load_weights(weights_path)
print(f"Loaded pretrained TRUE-FTT weights from {weights_path}")

for target_cell in [5, 7]:
    print(f"\n================= Fine-tuning on cell {target_cell} =================")
    df_c = df[df["cell"] == target_cell].copy()
    support_df = df_c.sample(n=SUPPORT_N, random_state=SEED)
    query_df   = df_c.drop(support_df.index)
    Xn_support, Xc_support, y_support = prep(support_df)
    Xn_query,   Xc_query,   y_query   = prep(query_df)

    model.compile(optimizer=Adam(learning_rate=FT_LR),
                  loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                           tf.keras.metrics.MeanSquaredError(name="mse"),
                           RootMeanSquaredError(name="rmse"),
                           R2Score(name="r2")])

    ckpt_path = os.path.join(OUT_DIR, f"best_epoch_cell{target_cell}.weights.h5")
    ckpt = ModelCheckpoint(ckpt_path, monitor="val_mae",
                           save_best_only=True, save_weights_only=True, mode="min", verbose=0)

    history = model.fit(
        (Xn_support, Xc_support), y_support,
        validation_data=((Xn_query, Xc_query), y_query),
        epochs=FT_EPOCHS, batch_size=FT_BATCH, verbose=1,
        callbacks=[QueryMetricsCallback(Xn_query, Xc_query, y_query), ckpt]
    )

    best_epoch = int(np.argmin(history.history["val_mae"]) + 1)
    print(f"Cell {target_cell} � Best epoch by val_mae: {best_epoch}")

    epochs_range = range(1, FT_EPOCHS + 1)

    plt.figure(figsize=(7,5))
    plt.plot(epochs_range, history.history["mae"], label="Support MAE", color="red")
    plt.plot(epochs_range, history.history["val_mae"], label="Test MAE", color="blue")
    plt.axvline(best_epoch, color="black", linestyle="--", label=f"Optimal Epoch={best_epoch}")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()
    plt.xlim(1, FT_EPOCHS)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ft_curve_MAE_support{SUPPORT_N}_epoch{best_epoch}_cell{target_cell}.pdf"))
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(epochs_range, history.history["mse"], label="Support MSE", color="red")
    plt.plot(epochs_range, history.history["val_mse"], label="Test MSE", color="blue")
    plt.axvline(best_epoch, color="black", linestyle="--", label=f"Optimal Epoch={best_epoch}")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
    plt.xlim(1, FT_EPOCHS)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ft_curve_MSE_support{SUPPORT_N}_epoch{best_epoch}_cell{target_cell}.pdf"))
    plt.show()

    model_best = FTTransformer(num_num=len(num_cols), num_cat=len(cat_cols), categories_cardinalities=cat_card)
    _ = model_best((tf.zeros((1, len(num_cols)), dtype=tf.float32),
                    tf.zeros((1, len(cat_cols)), dtype=tf.int32)))
    model_best.load_weights(ckpt_path)

    y_sup_pred = scaler_y.inverse_transform(model_best.predict((Xn_support, Xc_support), verbose=0))
    y_sup_true = scaler_y.inverse_transform(y_support)
    print("\n Support Set (Best Epoch)")
    print("MAE:", mean_absolute_error(y_sup_true, y_sup_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_sup_true, y_sup_pred)))
    print("R²:", r2_score(y_sup_true, y_sup_pred))

    t0 = time.time()
    y_q_pred = scaler_y.inverse_transform(model_best.predict((Xn_query, Xc_query), verbose=0))
    infer_per = (time.time() - t0) / len(Xn_query)
    y_q_true = scaler_y.inverse_transform(y_query)
    mae = mean_absolute_error(y_q_true, y_q_pred)
    rmse = np.sqrt(mean_squared_error(y_q_true, y_q_pred))
    r2  = r2_score(y_q_true, y_q_pred)
    print("\n Query Set (Best Epoch)")
    print("MAE:", mae); print("RMSE:", rmse); print("R²:", r2)
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
        "MAE": [mae],
        "RMSE": [rmse],
        "R2": [r2],
        "Inference/sample (s)": [infer_per],
        "Best Epoch": [best_epoch],
        "Support N": [SUPPORT_N],
    }
    out_xlsx = os.path.join(OUT_DIR, f"query_best_epoch_results_support{SUPPORT_N}_cell{target_cell}.xlsx")
    pd.DataFrame(results_dict).to_excel(out_xlsx, index=False)
    print(f"Saved Query Set best epoch results to {out_xlsx}")

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

    df_extra = pd.DataFrame([extra_results]); df_extra.index = ["R2"]
    extra_xlsx = os.path.join(OUT_DIR, f"query_additional_R2_support{SUPPORT_N}_cell{target_cell}.xlsx")
    df_extra.to_excel(extra_xlsx)
    print(f"Saved additional R² results to {extra_xlsx}")

