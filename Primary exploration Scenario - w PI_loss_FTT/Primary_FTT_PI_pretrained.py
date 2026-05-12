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
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

DATA_PATH = "final_combined_data.xlsx"
RESULT_DIR = "alpha_sweep_results"
os.makedirs(RESULT_DIR, exist_ok=True)

MARGIN = 0.0
ALPHA_LIST = [0.01, 0.1, 1, 10, 100]

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 600,
    "savefig.dpi": 600
})

df = pd.read_excel(DATA_PATH)
df = df[["t1","t2","d","cell","property","poisson_ratio"]].dropna().copy()
df["cell"] = pd.to_numeric(df["cell"], errors="coerce").astype(int)

num_cols = ["t1","t2","d","property", "cell"]  
cat_cols = ["cell"]                            
feat_cols = ["t1","t2","d","cell","property"]  

all_cats = sorted(df["cell"].unique().tolist())
cat_to_idx = {c: i for i, c in enumerate(all_cats)}
cat_card = [len(all_cats)]

cell_num_idx = num_cols.index("cell")

train_list, val_list = [], []
for c in [1,3,5]:
    df_c = df[df["cell"]==c].copy()
    y = df_c["poisson_ratio"].values
    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    tr, val = train_test_split(df_c, test_size=0.2, stratify=y_bins, random_state=SEED)
    train_list.append(tr); val_list.append(val)
train_df = pd.concat(train_list).reset_index(drop=True)
val_df   = pd.concat(val_list).reset_index(drop=True)

scaler_X_num = StandardScaler().fit(train_df[num_cols].values)
scaler_y = StandardScaler().fit(train_df["poisson_ratio"].values.reshape(-1,1))

def prep(df_):
    X_num = scaler_X_num.transform(df_[num_cols].values).astype(np.float32)
    X_cat = df_["cell"].map(cat_to_idx).values.astype(np.int32).reshape(-1,1)
    y = scaler_y.transform(df_["poisson_ratio"].values.reshape(-1,1)).astype(np.float32)
    return X_num, X_cat, y

Xn_train, Xc_train, y_train = prep(train_df)
Xn_val,   Xc_val,   y_val   = prep(val_df)

class MonotonicFTTransformer(Model):
    def __init__(self, num_num, num_cat, categories_cardinalities,
                 cell_num_col_idx, margin=0.0, alpha=1.0,
                 d_model=256, num_heads=4, num_layers=3, ff_dim=128, dropout=0.1):
        super().__init__()
        self.num_num = num_num
        self.num_cat = num_cat
        self.d_model = d_model
        self.cell_num_col_idx = int(cell_num_col_idx)
        self.margin = float(margin)
        self.alpha = float(alpha)

        self.num_linears = [Dense(d_model, use_bias=True, name=f"num_tok_{j}") for j in range(num_num)]

        self.cat_embs = []
        self.cat_bias = []
        for j, card in enumerate(categories_cardinalities):
            self.cat_embs.append(Embedding(input_dim=card, output_dim=d_model, name=f"cat_emb_{j}"))
            self.cat_bias.append(self.add_weight(
                name=f"cat_bias_{j}", shape=(d_model,), initializer="zeros", trainable=True
            ))

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

        self.mse_fn = tf.keras.losses.MeanSquaredError()

    def _forward_tokens(self, x_num, x_cat, training=False):
        B = tf.shape(x_num)[0]

        tok_num = []
        for j in range(self.num_num):
            col = tf.expand_dims(x_num[:, j], axis=1)        
            tok = self.num_linears[j](col)                 
            tok_num.append(tok)

        tok_cat = []
        for j in range(self.num_cat):
            e = self.cat_embs[j](x_cat[:, j])             
            e = e + self.cat_bias[j]
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


    def _mono_violation_loss(self, x_num, x_cat):
        with tf.GradientTape() as g:
            g.watch(x_num)
            y_ = self._forward_tokens(x_num, x_cat, training=True)
        J = g.batch_jacobian(y_, x_num)
        grads = tf.squeeze(J, axis=1)
        dY_dCell = grads[:, self.cell_num_col_idx]
        violation = tf.nn.relu(self.margin - dY_dCell)
        return tf.reduce_mean(tf.square(violation))

    def train_step(self, data):
        (x_num, x_cat), y = data
        with tf.GradientTape() as tape:
            y_pred = self._forward_tokens(x_num, x_cat, training=True)
            data_loss = self.mse_fn(y, y_pred)
            mono_loss = self._mono_violation_loss(x_num, x_cat)
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
    model_tmp = MonotonicFTTransformer(
        num_num=len(num_cols),
        num_cat=len(cat_cols),
        categories_cardinalities=cat_card,
        cell_num_col_idx=cell_num_idx,
        margin=MARGIN, alpha=alpha,
        d_model=256, num_heads=4, num_layers=3, ff_dim=128, dropout=0.1
    )

    model_tmp.compile(optimizer=Adam(1e-3),
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

    hist = model_tmp.fit((Xn_train, Xc_train), y_train,
                         validation_data=((Xn_val, Xc_val), y_val),
                         epochs=100, batch_size=64, verbose=0,
                         callbacks=[ckpt_tmp])

    model_tmp.load_weights(os.path.join(RESULT_DIR, f"tmp_best_alpha{alpha}.weights.h5"))
    yv_pred = model_tmp.predict((Xn_val, Xc_val), verbose=0)
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

model_best = MonotonicFTTransformer(
    num_num=len(num_cols),
    num_cat=len(cat_cols),
    categories_cardinalities=cat_card,
    cell_num_col_idx=cell_num_idx,
    margin=MARGIN, alpha=best_alpha,
    d_model=256, num_heads=4, num_layers=3, ff_dim=128, dropout=0.1
)
model_best.compile(optimizer=Adam(1e-3),
                   metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                            tf.keras.metrics.MeanSquaredError(name="mse")])

best_ckpt_path = os.path.join(RESULT_DIR, "ftt_pretrained.weights.h5")
ckpt = ModelCheckpoint(
    filepath=best_ckpt_path,
    monitor="val_mae",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=0
)

hist_best = model_best.fit((Xn_train, Xc_train), y_train,
                           validation_data=((Xn_val, Xc_val), y_val),
                           epochs=100, batch_size=64, verbose=0,
                           callbacks=[ckpt])

plt.figure(figsize=(7,5))
plt.plot(hist_best.history["mae"], label="Train MAE", color="red")
plt.plot(hist_best.history["val_mae"], label="Val MAE", color="blue")
plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()
plt.tick_params(axis="both", which="both", direction="in")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"best_learning_curve_MAE_alpha{best_alpha}.pdf"))
plt.show()

plt.figure(figsize=(7,5))
plt.plot(hist_best.history["mse"], label="Train MSE", color="red")
plt.plot(hist_best.history["val_mse"], label="Val MSE", color="blue")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
plt.tick_params(axis="both", which="both", direction="in")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"best_learning_curve_MSE_alpha{best_alpha}.pdf"))
plt.show()

assert os.path.exists(best_ckpt_path), "Best-epoch checkpoint not found."
model_best.load_weights(best_ckpt_path)

y_val_pred_best = model_best.predict((Xn_val, Xc_val), verbose=0)
y_val_true_inv = scaler_y.inverse_transform(y_val)
y_val_pred_inv = scaler_y.inverse_transform(y_val_pred_best)

mse_best = mean_squared_error(y_val_true_inv, y_val_pred_inv)
rmse_best = np.sqrt(mse_best)
mae_best = mean_absolute_error(y_val_true_inv, y_val_pred_inv)
r2_best  = r2_score(y_val_true_inv, y_val_pred_inv)
rho_best = spearmanr(y_val_true_inv.ravel(), y_val_pred_inv.ravel())[0]

print(f"\n Final (best alpha={best_alpha}) Best Epoch Val Performance:")
print(f"  MSE={mse_best:.6f}, RMSE={rmse_best:.6f}, "
      f"MAE={mae_best:.6f}, R²={r2_best:.4f}, Spearman={rho_best:.3f}")

final_results = pd.DataFrame([{
    "alpha": best_alpha,
    "MSE": mse_best,
    "RMSE": rmse_best,
    "MAE": mae_best,
    "R2": r2_best,
    "Spearman": rho_best
}])
final_results.to_excel(os.path.join(RESULT_DIR, "final_best_alpha_results.xlsx"), index=False)

plt.figure(figsize=(6,6))
plt.scatter(y_val_true_inv, y_val_pred_inv, c="#0000FF", s=15, alpha=0.5, edgecolors="none")
mn, mx = y_val_true_inv.min(), y_val_true_inv.max()
plt.plot([mn,mx],[mn,mx],"r-",lw=2,label="Ideal")
plt.xlabel("True"); plt.ylabel("Predicted"); plt.legend()
plt.tick_params(axis="both", which="both", direction="in")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"scatter_true_vs_pred_final_best_alpha{best_alpha}.pdf"))
plt.show()

metrics = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "R2": "R²"
}

for metric, title in metrics.items():
    plt.figure(figsize=(7,5))
    plt.plot(
        df_alpha["alpha"], df_alpha[metric],
        marker="o", linewidth=2,
        color="blue", markerfacecolor="blue", markeredgecolor="blue"
    )
    plt.xscale("log")
    plt.xlabel("Alpha (PI loss weight, log scale)")
    plt.ylabel(title)
    plt.grid(False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, f"alpha_vs_{metric.replace('/', '_')}_logx.pdf")
    plt.savefig(out_path)
    plt.show()

# ========== SAVE best_alpha ==========
with open(os.path.join(RESULT_DIR,"best_alpha.txt"), "w") as f:
    f.write(str(best_alpha))
print(f" Saved best model (by val_mae) and best_alpha.txt")
