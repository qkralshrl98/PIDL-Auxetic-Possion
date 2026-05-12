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
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Embedding  # Dropout 미사용
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

DATA_PATH = "final_combined_data.xlsx"
RESULT_DIR = "FTT_baseline_results"
os.makedirs(RESULT_DIR, exist_ok=True)

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

num_cols = ["t1","t2","d","property"]
cat_cols = ["cell"]  
feat_cols = num_cols + cat_cols

all_cats = sorted(df["cell"].unique().tolist())
cat_to_idx = {c: i for i, c in enumerate(all_cats)}
cat_card = [len(all_cats)]

train_list, val_list = [], []
for c in [1,3]:
    df_c = df[df["cell"] == c].copy()
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

class FTTransformer(Model):
    def __init__(self, num_num, num_cat, categories_cardinalities,
                 d_model=256, num_heads=4, num_layers=3, ff_dim=128, dropout=0.1):
        super().__init__()
        self.num_num = num_num
        self.num_cat = num_cat
        self.d_model = d_model
        self.num_linears = [Dense(d_model, use_bias=True, name=f"num_tok_{j}") for j in range(num_num)]
        self.cat_embs = []
        self.cat_bias = []
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
                LayerNormalization(epsilon=1e-6, name=f"enc{li}_ln2"),
                Dense(ff_dim, activation="relu", name=f"enc{li}_ff1"),
                Dense(d_model, name=f"enc{li}_ff2"),
            ])
        self.pred_norm = LayerNormalization(epsilon=1e-6, name="pred_ln")
        self.pred_dense = Dense(d_model, activation="relu", name="pred_dense")
        self.out = Dense(1, name="out")

    def call(self, inputs, training=False):
        x_num, x_cat = inputs
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

        for norm1, attn, norm2, ffd1, ffd2 in self.enc_layers:
            z_norm = norm1(z)
            attn_out = attn(z_norm, z_norm)
            z = z + attn_out

            z_norm2 = norm2(z)
            ff_out = ffd2(ffd1(z_norm2))
            z = z + ff_out

        cls_repr = z[:, 0, :]          
        h = self.pred_dense(self.pred_norm(cls_repr))
        return self.out(h)

model = FTTransformer(
    num_num=len(num_cols),
    num_cat=len(cat_cols),
    categories_cardinalities=cat_card,
    d_model=256, num_heads=4, num_layers=3, ff_dim=128, dropout=0.1
)

model.compile(optimizer=Adam(1e-4),
              loss="mse",
              metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                       tf.keras.metrics.MeanSquaredError(name="mse")])

ckpt = ModelCheckpoint(
    filepath=os.path.join(RESULT_DIR, "ftt_pretrained.h5"),
    monitor="val_mae",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=0
)

hist = model.fit((Xn_train, Xc_train), y_train,
                 validation_data=((Xn_val, Xc_val), y_val),
                 epochs=100, batch_size=64, verbose=1,
                 callbacks=[ckpt])

model.load_weights(os.path.join(RESULT_DIR, "ftt_pretrained.h5"))

y_val_pred = model.predict((Xn_val, Xc_val), verbose=0)
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
plt.plot([mn,mx],[mn,mx],"r-",lw=2) 
plt.xlabel("True Poisson's ratio"); plt.ylabel("Predicted Poisson's ratio")
plt.tick_params(axis="both", which="both", direction="in")

r2_text = f"R² = {r2:.4f}"
plt.text(0.05, 0.95, r2_text,
         transform=plt.gca().transAxes,
         fontsize=12, va="top", ha="left")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "scatter_true_vs_pred.pdf"))
plt.show()
