"""Stage 10 — monitoring dashboard (offline).

Run:  streamlit run dashboard/app.py

Reads results/<dataset>/summary.json and the saved models/processed data.
It does NOT capture live packets: it replays recorded test traffic through a
trained model to simulate monitoring, matching the project scope.
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config, processed_dir, results_dir
from src.models_neural import MLP, mlp_predict
from src.utils import load_json

st.set_page_config(page_title="IDS Adversarial Robustness", layout="wide")
cfg = load_config()

st.title("ML-IDS Adversarial Robustness — Monitoring Dashboard")

# --------------------------------------------------------------------------- #
# Dataset picker
# --------------------------------------------------------------------------- #
available = [d for d in cfg["datasets"]
             if os.path.exists(os.path.join(results_dir(cfg, d), "summary.json"))]
if not available:
    st.warning("No results yet. Run the pipeline first: `python scripts/run_all.py`")
    st.stop()

dataset = st.sidebar.selectbox("Dataset", available)
summary = load_json(os.path.join(results_dir(cfg, dataset), "summary.json"))


# --------------------------------------------------------------------------- #
# 1. Clean performance
# --------------------------------------------------------------------------- #
st.header("1 · Clean detection performance")
clean = summary.get("clean", {})
if clean:
    rows = []
    for model, m in clean.items():
        rows.append({
            "Model": model,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m.get("roc_auc"),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(4), use_container_width=True)


# --------------------------------------------------------------------------- #
# 2. Attack impact (BEFORE defense)
# --------------------------------------------------------------------------- #
st.header("2 · Attack impact (before defense)")
before = summary.get("attacks_before", {})
if before:
    rows = []
    for model, attacks in before.items():
        for atk, rec in attacks.items():
            rows.append({
                "Model": model, "Attack": atk,
                "ASR": rec["asr"],
                "Robustness": rec["robustness_score"],
                "Acc under attack": rec["acc_under_attack"],
            })
    df = pd.DataFrame(rows)
    st.dataframe(df.round(4), use_container_width=True)

    fig = go.Figure()
    for model in df["Model"].unique():
        sub = df[df["Model"] == model]
        fig.add_bar(name=model, x=sub["Attack"], y=sub["ASR"])
    fig.update_layout(barmode="group", title="Attack Success Rate by model and attack",
                      yaxis_title="ASR (higher = worse)")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 3. Defense effect (AFTER) — before vs after
# --------------------------------------------------------------------------- #
st.header("3 · Defense effect (before vs after)")
after = summary.get("after", {})
adv = after.get("adv_training_MLP", {})
if adv:
    rows = []
    for atk, rec in adv.items():
        undef = before.get("MLP", {}).get(atk, {})
        rows.append({
            "Attack": atk,
            "Acc under attack (no defense)": undef.get("acc_under_attack"),
            "Acc under attack (adv-trained)": rec["acc_under_attack"],
            "Defense Recovery Rate": rec.get("drr"),
        })
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Hardened clean detection (malicious)",
              f"{after.get('clean_malicious_acc_hardened', float('nan')):.3f}")
    c2.metric("Baseline clean detection (malicious)",
              f"{after.get('clean_malicious_acc_baseline', float('nan')):.3f}",
              help="Difference is the clean-accuracy trade-off of adversarial training.")

if after.get("unseen_attack_note"):
    st.caption(after["unseen_attack_note"])


# --------------------------------------------------------------------------- #
# 3.5 · Naive vs. consistency-preserving attack — the headline contrast
# --------------------------------------------------------------------------- #
st.header("3.5 · Naive vs. consistency-preserving attack (headline result)")
st.caption(
    "The same sanitizer (range clip + feature squeeze + recompute derived "
    "features from base features) is applied to two attack variants. A naive "
    "attack perturbs a derived feature (e.g. a mean) independently of the "
    "base features it is computed from; a consistency-preserving attack only "
    "touches base features and lets the derived ones follow. Expect the "
    "naive attack's ASR to collapse after sanitization while the consistent "
    "attack's ASR survives."
)
san_naive = after.get("sanitization_MLP_naive", after.get("sanitization_MLP", {}))
san_consistent = after.get("sanitization_MLP_consistent", {})
before_naive = before.get("MLP", {})
before_consistent = summary.get("attacks_before_consistent", {}).get("MLP", {})

if san_naive and san_consistent:
    rows = []
    for atk, rec in san_naive.items():
        rows.append({
            "Attack": atk, "Mode": "Naive",
            "ASR before sanitize": before_naive.get(atk, {}).get("asr"),
            "ASR after sanitize": rec["asr"],
        })
    for atk, rec in san_consistent.items():
        rows.append({
            "Attack": atk, "Mode": "Consistent",
            "ASR before sanitize": before_consistent.get(atk, {}).get("asr"),
            "ASR after sanitize": rec["asr"],
        })
    df_contrast = pd.DataFrame(rows)
    st.dataframe(df_contrast.round(4), use_container_width=True)

    fig3 = go.Figure()
    for mode in ["Naive", "Consistent"]:
        sub = df_contrast[df_contrast["Mode"] == mode]
        fig3.add_bar(name=f"{mode} — after sanitize", x=sub["Attack"], y=sub["ASR after sanitize"])
    fig3.update_layout(
        barmode="group",
        title="ASR after sanitization: naive (defeated) vs. consistent (survives)",
        yaxis_title="ASR (higher = attack still evades)",
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Run Stage 6 and Stage 8 (`python scripts/04_generate_attacks.py` then "
            "`python scripts/06_reevaluate.py`) to populate this contrast.")


# --------------------------------------------------------------------------- #
# 4. Offline traffic replay
# --------------------------------------------------------------------------- #
st.header("4 · Replay recorded test traffic")
st.caption("Offline replay of recorded test flows — not live capture.")

mdir = os.path.join(results_dir(cfg, dataset), "models")
pdir = processed_dir(cfg, dataset)
model_choice = st.selectbox("Model", ["RandomForest", "XGBoost", "SVM", "MLP"])
n_replay = st.slider("Flows to replay", 50, 1000, 300, step=50)
speed = st.slider("Chunk size", 10, 200, 50, step=10)

if st.button("Start replay"):
    x_test = np.load(os.path.join(pdir, "x_test.npy"))
    y_test = np.load(os.path.join(pdir, "y_test.npy"))
    idx = np.random.default_rng(cfg["seed"]).permutation(len(x_test))[:n_replay]
    xr, yr = x_test[idx], y_test[idx]

    if model_choice == "MLP":
        arch = load_json(os.path.join(mdir, "MLP_arch.json"))
        model = MLP(arch["in_dim"], arch["n_classes"], tuple(arch["hidden"]), arch["dropout"])
        import torch
        model.load_state_dict(torch.load(os.path.join(mdir, "MLP.pt"), map_location="cpu"))
        model.eval()
        predict = lambda x: mlp_predict(model, x)
    else:
        model = joblib.load(os.path.join(mdir, f"{model_choice}.joblib"))
        predict = model.predict

    prog = st.progress(0.0)
    log_box = st.empty()
    tp = fp = tn = fn = 0
    for start in range(0, len(xr), speed):
        chunk_x = xr[start:start + speed]
        chunk_y = yr[start:start + speed]
        pred = predict(chunk_x)
        tp += int(((pred == 1) & (chunk_y == 1)).sum())
        fp += int(((pred == 1) & (chunk_y == 0)).sum())
        tn += int(((pred == 0) & (chunk_y == 0)).sum())
        fn += int(((pred == 0) & (chunk_y == 1)).sum())
        prog.progress(min(1.0, (start + speed) / len(xr)))
        log_box.write(
            f"Processed {min(start + speed, len(xr))}/{len(xr)} flows — "
            f"detected attacks: {tp} | missed attacks: {fn} | false alarms: {fp}"
        )
        time.sleep(0.05)

    st.success("Replay complete.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True detections", tp)
    m2.metric("Missed attacks", fn)
    m3.metric("False alarms", fp)
    m4.metric("Benign correct", tn)
