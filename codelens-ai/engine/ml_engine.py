"""
ML engine for CodeLens AI.

Follows the base paper (Leyton-Yela et al., 2026) experimental setup:
  * dataset of 3,000 labelled while-loop tasks (Gries 7-class scheme)
  * 80% / 20% stratified train-test split, seed = 2023
  * scikit-learn models with the paper's hyper-parameters:
        KNN : 15 neighbours, Manhattan distance
        SVM : RBF kernel, gamma=0.000379, C=233.572147
        RF  : 70 trees, max_depth=15
        MLP : adaptive learning rate, architecture [100,100,100]
        Vt  : soft-voting ensemble (RF + MLP)
        St  : stacking ensemble (RF + MLP + Logistic Regression)
  * metrics: Accuracy, AUC (macro OvR, of the precision-recall family used in
    the paper), MCC (Matthews correlation coefficient)
  * visualisations are produced with matplotlib (dark theme), matching the
    paper's bar charts (Fig. 7, 9, 10) and confusion matrix (Fig. 12).

Inputs to the models are the per-state AST feature vectors (see gries.py),
the structural analogue of the paper's Gries-partition embeddings.
"""
import io
import os
import pickle
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)

from engine.gries import feature_vector, FEATURE_NAMES

SEED = 2023
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "..", "data", "gries_loop_tasks.csv")
MODEL_PATH = os.path.join(HERE, "..", "data", "model_bundle.pkl")

CLASS_NAMES = ["ys - Initial state", "ye - Final state", "yt - State transformation",
               "yst - Initial+Transformation", "yse - Initial+Final",
               "yet - Final+Transformation", "yste - Initial+Final+Transformation"]
CLASS_SHORT = ["S (ys)", "E (ye)", "T (yt)", "S+T (yst)", "S+E (yse)",
               "E+T (yet)", "S+T+E (yste)"]

# ---------------------------------------------------------------------------
# dark theme (matches the website)
# ---------------------------------------------------------------------------
BG = "#0f1226"
PANEL = "#1a1f3d"
GRID = "#2c3260"
COLORS = ["#7c5cff", "#00d4ff", "#ff5ca8", "#ffc24b", "#3ddc97", "#ff7b54", "#a78bfa"]


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#c7cbe8", labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels():
        lbl.set_color("#c7cbe8")


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def load_matrix():
    df = pd.read_csv(DATA_CSV)
    X = np.array([feature_vector(code) for code in df["Solution"]], dtype=float)
    y = df["Label_Code"].to_numpy(dtype=int)
    return X, y, df


def build_models():
    knn = Pipeline([("sc", StandardScaler()),
                    ("m", KNeighborsClassifier(n_neighbors=15, p=1, weights="distance"))])
    svm = Pipeline([("sc", StandardScaler()),
                    ("m", SVC(kernel="rbf", gamma=0.000379, C=233.572147,
                              probability=True, random_state=SEED))])
    rf = RandomForestClassifier(n_estimators=70, max_depth=15, random_state=SEED, n_jobs=-1)
    mlp = Pipeline([("sc", StandardScaler()),
                    ("m", MLPClassifier(hidden_layer_sizes=(100, 100, 100),
                                        learning_rate="adaptive", max_iter=400,
                                        random_state=SEED))])
    vote = VotingClassifier(estimators=[("rf", rf), ("mlp", mlp)], voting="soft")
    stack = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=70, max_depth=15,
                                                  random_state=SEED, n_jobs=-1)),
                    ("mlp", Pipeline([("sc", StandardScaler()),
                                      ("m", MLPClassifier(hidden_layer_sizes=(100, 100, 100),
                                                          learning_rate="adaptive",
                                                          max_iter=400,
                                                          random_state=SEED))])),
                    ("lr", Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=500,
                                                              random_state=SEED))]))],
        final_estimator=LogisticRegression(max_iter=500, random_state=SEED),
        stack_method="predict_proba", n_jobs=-1)
    return {"KNN": knn, "SVM": svm, "RF": rf, "MLP": mlp, "Voting (RF+MLP)": vote,
            "Stacking (RF+MLP+LR)": stack}


def train(verbose=True):
    X, y, df = load_matrix()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y,
                                          random_state=SEED)
    models = build_models()
    rows = []
    fitted = {}
    for name, model in models.items():
        if verbose:
            print(f"  training {name} ...", flush=True)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)
        acc = accuracy_score(yte, pred)
        mcc = matthews_corrcoef(yte, pred)
        try:
            auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                                labels=list(range(7)))
        except ValueError:
            auc = 0.0
        rows.append({"model": name, "accuracy": acc, "auc": auc, "mcc": mcc})
        fitted[name] = model

    bundle = {"models": fitted, "metrics": pd.DataFrame(rows),
              "feature_names": FEATURE_NAMES, "class_names": CLASS_SHORT,
              "y_test": yte, "pred_vote": fitted["Voting (RF+MLP)"].predict(Xte),
              "df": df}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    if verbose:
        print(bundle["metrics"].to_string(index=False))
    return bundle


def get_bundle():
    if not os.path.exists(MODEL_PATH):
        return train()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# prediction for user code
# ---------------------------------------------------------------------------
def predict_code(code):
    bundle = get_bundle()
    model = bundle["models"]["Voting (RF+MLP)"]
    vec = np.array([feature_vector(code)], dtype=float)
    proba = model.predict_proba(vec)[0]
    classes = list(range(7))
    order = np.argsort(proba)[::-1]
    return {
        "prediction_code": int(model.predict(vec)[0]),
        "prediction": CLASS_SHORT[int(model.predict(vec)[0])],
        "probabilities": {CLASS_SHORT[c]: float(proba[c]) for c in classes},
        "ranking": [{"label": CLASS_SHORT[c], "prob": float(proba[c])} for c in order],
    }


# ---------------------------------------------------------------------------
# matplotlib charts (paper style, dark theme) -> base64 png
# ---------------------------------------------------------------------------
def chart_model_comparison(bundle):
    df = bundle["metrics"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    x = np.arange(len(df))
    w = 0.26
    bars = []
    for i, (metric, color) in enumerate([("accuracy", COLORS[0]),
                                         ("auc", COLORS[1]),
                                         ("mcc", COLORS[2])]):
        b = ax.bar(x + (i - 1) * w, df[metric], w, label=metric.upper(),
                   color=color, edgecolor="none", alpha=0.92)
        bars.append(b)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color="#e6e8ff")
    ax.set_title("ML model comparison  —  Accuracy / AUC / MCC  (paper Table I & Fig. 9)",
                 color="#ffffff", fontsize=11, pad=12, weight="bold")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor="#e6e8ff", ncol=3,
                    loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False)
    for b in bars:
        for r in b:
            r.set_linewidth(0)
    return _fig_b64(fig)


def chart_label_distribution(bundle):
    df = bundle["df"]
    counts = df.groupby("Label_Code").size().reindex(range(7), fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    bars = ax.bar(range(7), counts.values, color=COLORS, alpha=0.92, width=0.68)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v}\n{v/len(df)*100:.1f}%",
                ha="center", va="bottom", color="#e6e8ff", fontsize=8.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(CLASS_SHORT, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("Records", color="#e6e8ff")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.set_title("Dataset distribution by Gries error state (paper: 29% S, 29% E, 29% T, 4% pairs, 3% all)",
                 color="#ffffff", fontsize=10.5, pad=12, weight="bold")
    return _fig_b64(fig)


def chart_confusion_matrix(bundle):
    cm = confusion_matrix(bundle["y_test"], bundle["pred_vote"], labels=range(7))
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    im = ax.imshow(cm, cmap="magma")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(["ys", "ye", "yt", "yst", "yse", "yet", "yste"],
                       color="#c7cbe8", fontsize=9)
    ax.set_yticklabels(["ys", "ye", "yt", "yst", "yse", "yet", "yste"],
                       color="#c7cbe8", fontsize=9)
    ax.set_xlabel("Predicted label", color="#e6e8ff")
    ax.set_ylabel("True label", color="#e6e8ff")
    total = cm.sum()
    for i in range(7):
        for j in range(7):
            v = cm[i, j]
            if v:
                color = "white" if v < cm.max() * 0.55 else "black"
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Confusion matrix — Voting(RF+MLP) model (paper Fig. 12)",
                 color="#ffffff", fontsize=10.5, pad=12, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.yaxis.set_tick_params(color="#c7cbe8")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#c7cbe8")
    return _fig_b64(fig)


def chart_feature_importance(bundle):
    rf = bundle["models"]["RF"]
    names = bundle["feature_names"]
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1][:14]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    vals = imp[order][::-1]
    labels = [names[i] for i in order][::-1]
    cols = [COLORS[0] if l.startswith("S_") else COLORS[1] if l.startswith("T_")
            else COLORS[2] if l.startswith("E_") else COLORS[3] for l in labels]
    ax.barh(range(len(vals)), vals, color=cols, alpha=0.92)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Random Forest importance", color="#e6e8ff")
    ax.set_title("Top AST/Gries-state features used by the ML classifier",
                 color="#ffffff", fontsize=10.5, pad=12, weight="bold")
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[Patch(color=COLORS[0], label="S state"),
                             Patch(color=COLORS[1], label="T state"),
                             Patch(color=COLORS[2], label="E state"),
                             Patch(color=COLORS[3], label="global")],
                    facecolor=PANEL, edgecolor=GRID, labelcolor="#e6e8ff",
                    loc="lower right", frameon=False)
    return _fig_b64(fig)


def chart_state_features(feats):
    """Bar chart of AST node counts per Gries state for one code snippet."""
    metrics = ["lines", "Assign", "AugAdd", "BinAdd", "BinMod", "CmpLtE", "CmpGtE",
               "If", "Call", "Const"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    x = np.arange(len(metrics))
    w = 0.26
    for i, (st, color) in enumerate([("S", COLORS[0]), ("T", COLORS[1]), ("E", COLORS[2])]):
        vals = [feats.get(f"{st}_{m}", 0) for m in metrics]
        ax.bar(x + (i - 1) * w, vals, w, label=f"{st} state", color=color, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(["lines", "assign", "aug +=", "bin +", "bin %",
                        "cmp <=", "cmp >=", "if", "calls", "consts"],
                       rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("AST node count", color="#e6e8ff")
    ax.set_title("AST feature profile across Gries states (S / T / E) of your code",
                 color="#ffffff", fontsize=10.5, pad=12, weight="bold")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor="#e6e8ff", frameon=False)
    return _fig_b64(fig)


def chart_prediction_proba(ranking):
    labels = [r["label"] for r in ranking][::-1]
    probs = [r["prob"] for r in ranking][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    cols = [COLORS[i % len(COLORS)] for i in range(len(labels))][::-1]
    bars = ax.barh(range(len(labels)), probs, color=cols, alpha=0.92)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Predicted probability", color="#e6e8ff")
    ax.set_title("ML classification probabilities over the 7 Gries error labels",
                 color="#ffffff", fontsize=10.5, pad=12, weight="bold")
    for b, p in zip(bars, probs):
        ax.text(min(p + 0.02, 1.0), b.get_y() + b.get_height() / 2, f"{p*100:.1f}%",
                va="center", color="#e6e8ff", fontsize=8.5)
    return _fig_b64(fig)


if __name__ == "__main__":
    print("Training models (seed 2023, stratified 80/20) ...")
    train()
    print("Bundle saved ->", MODEL_PATH)
