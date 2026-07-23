"""Professional evaluation dashboard for the adversarial NIDS project.

Run from the repository root:
    streamlit run dashboard/app.py

The dashboard is read-only. Every result is loaded from summary.json; values
are never generated or silently replaced with demo data.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODEL_NAMES = {
    "rf": "Random Forest",
    "randomforest": "Random Forest",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
    "svm": "Support Vector Machine",
    "mlp": "Multilayer Perceptron",
}
MODEL_COLORS = {
    "Random Forest": "#2DD4BF",
    "XGBoost": "#60A5FA",
    "Support Vector Machine": "#FBBF24",
    "Multilayer Perceptron": "#A78BFA",
}
PORT_NAMES = {
    20: "FTP Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS",
    445: "SMB", 993: "IMAPS", 995: "POP3S", 1433: "MSSQL",
    3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL", 5900: "VNC",
    8080: "HTTP Alt", 8443: "HTTPS Alt",
}


st.set_page_config(
    page_title="Aegis | Adversarial NIDS Evaluation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #080d18; }
    .block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] { background: #0c1424; border-right: 1px solid #1e293b; }
    h1, h2, h3 { letter-spacing: -0.025em; text-wrap: balance; }
    h1 { font-size: clamp(1.7rem, 3vw, 2.4rem) !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #101a2d, #0d1627);
        border: 1px solid #22304a;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 126px;
    }
    [data-testid="stMetricLabel"] { color: #9aa9c2; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    [data-testid="stDataFrame"] { border: 1px solid #22304a; border-radius: 10px; }
    .eyebrow { color: #5eead4; font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .lede { color: #9aa9c2; max-width: 780px; font-size: .98rem; line-height: 1.65; }
    .status-chip {
        display: inline-block; border: 1px solid #285548; background: #0c2b25;
        color: #86efac; border-radius: 999px; padding: .25rem .65rem;
        font-size: .74rem; font-weight: 650;
    }
    .note {
        border-left: 3px solid #38bdf8; background: #0d1a2b; color: #b9c6d9;
        border-radius: 0 8px 8px 0; padding: .85rem 1rem; margin: .5rem 0 1rem;
    }
    .definition { color: #93a4bd; font-size: .84rem; line-height: 1.55; }
    div[data-baseweb="tab-list"] { gap: .35rem; }
    button[data-baseweb="tab"] { border-radius: 8px; padding-inline: 1rem; }
    button:focus-visible, a:focus-visible { outline: 2px solid #67e8f9 !important; outline-offset: 2px; }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def finite(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def percent(value: Any, digits: int = 1) -> str:
    number = finite(value)
    return "N/A" if number is None else f"{number:.{digits}%}"


def model_label(value: Any) -> str:
    raw = str(value)
    return MODEL_NAMES.get(raw.lower(), raw)


def prettify_attack(value: Any) -> str:
    return str(value).replace("_", " ").replace("(", " (").strip().title().replace("Pgd", "PGD").replace("Fgsm", "FGSM").replace("Jsma", "JSMA")


@st.cache_data(show_spinner=False)
def read_summary(path_string: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns  # included only to invalidate the cache when the file changes
    with Path(path_string).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("The summary root must be a JSON object.")
    return data


def discover_summaries() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*/summary.json")) if RESULTS_DIR.exists() else []


def clean_frame(summary: dict[str, Any]) -> pd.DataFrame:
    records = summary.get("clean_metrics") or summary.get("clean") or {}
    rows: list[dict[str, Any]] = []
    for model, values in records.items():
        if not isinstance(values, dict):
            continue
        matrix = values.get("confusion_matrix")
        tn = fp = fn = tp = None
        if isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix):
            tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        rows.append({
            "Model": model_label(model),
            "Accuracy": finite(values.get("accuracy")),
            "Precision": finite(values.get("precision")),
            "Recall / TPR": finite(values.get("recall")),
            "F1-Score": finite(values.get("f1")),
            "ROC-AUC": finite(values.get("roc_auc")),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    return pd.DataFrame(rows)


def attack_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flat = summary.get("attack_results")
    has_flat_records = isinstance(flat, list) and any(isinstance(item, dict) for item in flat)
    if isinstance(flat, list):
        for item in flat:
            if not isinstance(item, dict):
                continue
            rows.append({
                "Model": model_label(item.get("model", "Unknown")),
                "Attack": prettify_attack(item.get("attack", "Unknown")),
                "Threat Model": item.get("threat_model", "Not recorded"),
                "Epsilon": finite(item.get("perturbation_budget")),
                "Samples": item.get("n_attacked"),
                "Valid Samples": item.get("n_valid"),
                "Validity Rate": (finite(item.get("n_valid")) / finite(item.get("n_attacked"))) if finite(item.get("n_valid")) is not None and finite(item.get("n_attacked")) else None,
                "Raw ASR": finite(item.get("raw_attack_success")),
                "Valid ASR": finite(item.get("valid_attack_success", item.get("asr"))),
                "Constraint": "Consistent" if "consistent" in str(item.get("attack", "")).lower() else ("Naive" if "naive" in str(item.get("attack", "")).lower() else "Constrained"),
            })
    for section, constraint in (("attacks_before", "Naive"), ("attacks_before_consistent", "Consistent")):
        nested = summary.get(section)
        if not isinstance(nested, dict) or has_flat_records:
            continue
        for model, attacks in nested.items():
            if not isinstance(attacks, dict):
                continue
            for attack, item in attacks.items():
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "Model": model_label(model), "Attack": prettify_attack(attack),
                    "Threat Model": item.get("threat_model", "Not recorded"),
                    "Epsilon": finite(item.get("epsilon")), "Samples": item.get("n_attacked"),
                    "Valid Samples": item.get("n_valid"), "Validity Rate": finite(item.get("validity_rate")),
                    "Raw ASR": finite(item.get("raw_asr", item.get("asr"))),
                    "Valid ASR": finite(item.get("asr")), "Constraint": constraint,
                })
    return pd.DataFrame(rows)


def defense_frame(summary: dict[str, Any]) -> pd.DataFrame:
    table = summary.get("robustness_table")
    rows: list[dict[str, Any]] = []
    if isinstance(table, list):
        for item in table:
            if not isinstance(item, dict):
                continue
            clean = finite(item.get("clean_metric_accuracy"))
            undefended = finite(item.get("undefended_attacked_accuracy"))
            defended = finite(item.get("defended_attacked_accuracy"))
            rows.append({
                "Model": model_label(item.get("model", "Unknown")),
                "Evaluation Attack": prettify_attack(item.get("attack", "Unknown")),
                "Clean Accuracy": clean,
                "Undefended Accuracy": undefended,
                "Defended Accuracy": defended,
                "Robustness Before": undefended / clean if clean and undefended is not None else None,
                "Robustness After": defended / clean if clean and defended is not None else None,
                "DRR": finite(item.get("defense_recovery_rate", item.get("drr"))),
                "Clean Cost": finite(item.get("clean_accuracy_cost_of_defense")),
            })
    after = summary.get("after", {})
    clean = summary.get("clean", {})
    adv = after.get("adv_training_MLP", {}) if isinstance(after, dict) else {}
    if not rows and isinstance(adv, dict):
        clean_acc = finite(clean.get("MLP", {}).get("accuracy")) if isinstance(clean.get("MLP"), dict) else None
        for attack, item in adv.items():
            if isinstance(item, dict):
                rows.append({
                    "Model": "Multilayer Perceptron", "Evaluation Attack": prettify_attack(attack),
                    "Clean Accuracy": clean_acc, "Undefended Accuracy": finite(item.get("acc_before")),
                    "Defended Accuracy": finite(item.get("acc_after")), "Robustness Before": finite(item.get("robustness_before")),
                    "Robustness After": finite(item.get("robustness")), "DRR": finite(item.get("drr")), "Clean Cost": None,
                })
    return pd.DataFrame(rows)


def chart_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"), margin=dict(l=10, r=10, t=35, b=20),
        legend_title_text="", hoverlabel=dict(bgcolor="#101a2d"),
    )
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")
    return fig


def styled_table(frame: pd.DataFrame, percent_columns: list[str], height: int | None = None) -> None:
    formats = {column: st.column_config.NumberColumn(column, format="%.2f%%") for column in percent_columns if column in frame.columns}
    table_options: dict[str, Any] = {
        "hide_index": True,
        "use_container_width": True,
        "column_config": formats,
    }
    if height is not None:
        table_options["height"] = height
    st.dataframe(frame, **table_options)


def port_label(value: Any) -> str:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return str(value)
    service = PORT_NAMES.get(port)
    return f"{port} · {service}" if service else str(port)


@st.cache_data(show_spinner=False)
def generate_demo_traffic(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate presentation-only SOC events; never used as research evidence."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    countries = [
        ("China", 24), ("United States", 18), ("Russia", 14), ("India", 10),
        ("Brazil", 7), ("Germany", 6), ("Netherlands", 5), ("France", 5),
        ("United Kingdom", 4), ("South Korea", 3), ("Iran", 2), ("Vietnam", 2),
    ]
    attack_types = ["Benign", "DDoS", "DoS", "Port Scan", "Brute Force", "Botnet", "Web Attack", "Infiltration"]
    common_ports = [80, 443, 22, 53, 3389, 8080, 21, 25, 445, 1433, 3306, 8443]
    protocols = ["TCP", "UDP", "HTTP", "HTTPS", "ICMP", "DNS", "SSH"]
    source_pool = [f"{rng.integers(1, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(80)]
    now = datetime.now()
    rows = []
    for _ in range(n_rows):
        label = random.choices(attack_types, weights=[56, 10, 9, 8, 7, 4, 4, 2])[0]
        malicious = label != "Benign"
        severity = random.choices(
            ["Critical", "High", "Medium", "Low", "Info"],
            weights=[18, 34, 31, 12, 5] if malicious else [0, 1, 8, 36, 55],
        )[0]
        action = random.choices(["Blocked", "Alerted", "Allowed"], weights=[61, 29, 10])[0] if malicious else "Allowed"
        country = random.choices([item[0] for item in countries], weights=[item[1] for item in countries])[0]
        rows.append({
            "timestamp": now - timedelta(seconds=int(rng.integers(0, 36 * 3600))),
            "src_ip": random.choice(source_pool),
            "dst_ip": f"10.{rng.integers(0, 16)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
            "src_port": int(rng.integers(1024, 65536)),
            "dst_port": random.choice(common_ports) if random.random() < .82 else int(rng.integers(1024, 65536)),
            "protocol": random.choice(protocols), "packet_length": int(np.clip(rng.lognormal(6.2, .8), 40, 1500)),
            "label": label, "severity": severity, "action": action, "country": country,
        })
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)


def normalize_traffic_csv(source: pd.DataFrame) -> pd.DataFrame:
    """Map common flow-log column names to the dashboard's operational schema."""
    aliases = {
        "timestamp": ["timestamp", "time", "datetime", "date", "ts"],
        "src_ip": ["src_ip", "source_ip", "srcip", "source ip", "src"],
        "dst_ip": ["dst_ip", "destination_ip", "dstip", "destination ip", "dst"],
        "src_port": ["src_port", "source_port", "sport"],
        "dst_port": ["dst_port", "destination_port", "dport", "destination port", "port"],
        "protocol": ["protocol", "proto"], "packet_length": ["packet_length", "pkt_len", "length", "bytes"],
        "label": ["label", "attack", "class", "category", "attack_cat"],
        "severity": ["severity", "priority", "risk"], "action": ["action", "verdict"],
        "country": ["country", "src_country", "source_country"],
    }
    lookup = {str(column).lower().strip(): column for column in source.columns}
    def find(name: str) -> Any:
        return next((lookup[item] for item in aliases[name] if item in lookup), None)
    defaults = {
        "src_ip": "Unknown", "dst_ip": "Unknown", "src_port": 0, "dst_port": 0,
        "protocol": "Unknown", "packet_length": 0, "label": "Unknown",
        "severity": "Medium", "action": "Observed", "country": "Unknown",
    }
    output = pd.DataFrame(index=source.index)
    timestamp_column = find("timestamp")
    output["timestamp"] = pd.to_datetime(source[timestamp_column], errors="coerce") if timestamp_column else pd.Timestamp.now()
    for name, default in defaults.items():
        column = find(name)
        output[name] = source[column] if column else default
    for name in ["src_port", "dst_port", "packet_length"]:
        output[name] = pd.to_numeric(output[name], errors="coerce").fillna(0).astype(int)
    output["protocol"] = output["protocol"].astype(str).str.upper()
    output["label"] = output["label"].astype(str)
    output["severity"] = output["severity"].astype(str).str.title()
    output["action"] = output["action"].astype(str).str.title()
    output["country"] = output["country"].astype(str)
    return output.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False).reset_index(drop=True)


summaries = discover_summaries()
with st.sidebar:
    st.markdown("## 🛡️ Aegis NIDS")
    st.caption("Adversarial robustness evaluation console")
    st.divider()
    if summaries:
        selected = st.selectbox(
            "Experiment Dataset",
            summaries,
            format_func=lambda path: path.parent.name.replace("_", " ").upper(),
            help="Each option maps to results/<dataset>/summary.json.",
        )
    else:
        selected = None
        st.error("No experiment summary found.")
    st.divider()
    st.markdown("**Evaluation Definitions**")
    st.markdown(
        '<p class="definition"><b>ASR</b> — detected malicious flows that evade after attack.<br><br>'
        '<b>Robustness</b> — attacked accuracy divided by clean accuracy.<br><br>'
        '<b>DRR</b> — fraction of attack-induced loss recovered by a defense.</p>',
        unsafe_allow_html=True,
    )
    st.caption("All values are read from the selected experiment artifact.")

st.markdown('<div class="eyebrow">Major Project · Evaluation Dashboard</div>', unsafe_allow_html=True)
st.title("Adversarial Robustness of ML-Based NIDS")
st.markdown(
    '<p class="lede">Evidence-focused reporting for baseline detection, realistic constrained evasion, '
    'and defense recovery across machine-learning intrusion detectors.</p>',
    unsafe_allow_html=True,
)

if selected is None:
    st.info("Run the experiment pipeline first. Expected output: `results/<dataset>/summary.json`.")
    st.stop()

try:
    summary = read_summary(str(selected), selected.stat().st_mtime_ns)
except (OSError, json.JSONDecodeError, ValueError) as exc:
    st.error(f"Could not load the selected summary: {exc}")
    st.stop()

clean_df = clean_frame(summary)
attack_df = attack_frame(summary)
defense_df = defense_frame(summary)
dataset_name = str(summary.get("dataset") or selected.parent.name).upper()
status = str(summary.get("status", "recorded")).title()

st.markdown(f'<span class="status-chip">● {status}</span>&nbsp;&nbsp; **{dataset_name}** · Seed `{summary.get("seed", "not recorded")}`', unsafe_allow_html=True)

best_clean = clean_df.loc[clean_df["F1-Score"].idxmax()] if not clean_df.empty and clean_df["F1-Score"].notna().any() else None
worst_attack = attack_df.loc[attack_df["Valid ASR"].idxmax()] if not attack_df.empty and attack_df["Valid ASR"].notna().any() else None
best_defense = defense_df.loc[defense_df["DRR"].idxmax()] if not defense_df.empty and defense_df["DRR"].notna().any() else None
validity = attack_df["Validity Rate"].mean() if not attack_df.empty and "Validity Rate" in attack_df else None

st.markdown("### Evaluation Snapshot")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best Baseline F1", percent(best_clean["F1-Score"] if best_clean is not None else None), best_clean["Model"] if best_clean is not None else "No clean metrics")
k2.metric("Highest Valid ASR", percent(worst_attack["Valid ASR"] if worst_attack is not None else None), f'{worst_attack["Model"]} · {worst_attack["Attack"]}' if worst_attack is not None else "No attack metrics", delta_color="inverse")
k3.metric("Best Defense Recovery", percent(best_defense["DRR"] if best_defense is not None else None), best_defense["Model"] if best_defense is not None else "No defense metrics")
k4.metric("Mean Attack Validity", percent(validity), "Constraint-compliant samples")

overview_tab, baseline_tab, attack_tab, defense_tab, matrix_tab, operations_tab = st.tabs([
    "Executive Overview", "Baseline Models", "Attack Evaluation", "Defense Evaluation", "Evaluation Matrix", "NIDS Operations"
])

with overview_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Model Quality Profile")
        if clean_df.empty:
            st.info("Clean classification metrics are not available.")
        else:
            radar_metrics = ["Accuracy", "Precision", "Recall / TPR", "F1-Score", "ROC-AUC"]
            fig = go.Figure()
            for _, row in clean_df.iterrows():
                values = [row[item] or 0 for item in radar_metrics]
                fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=radar_metrics + [radar_metrics[0]], fill="toself", name=row["Model"], opacity=.72))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#334155"), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(chart_layout(fig, 420), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Top 5 Attack Scenarios by Valid ASR")
        if attack_df.empty:
            st.info("Attack metrics are not available.")
        else:
            top5 = attack_df.nlargest(5, "Valid ASR").sort_values("Valid ASR")
            top5 = top5.assign(Scenario=top5["Model"] + " · " + top5["Attack"])
            fig = px.bar(top5, x="Valid ASR", y="Scenario", orientation="h", color="Constraint", text="Valid ASR", color_discrete_map={"Consistent": "#F97316", "Naive": "#60A5FA", "Constrained": "#A78BFA"})
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
            fig.update_xaxes(tickformat=".0%", range=[0, min(1.05, max(.1, float(top5["Valid ASR"].max()) * 1.18))], title="Valid Attack Success Rate")
            st.plotly_chart(chart_layout(fig, 420), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="note"><b>Defense interpretation:</b> prioritize scenarios with high valid ASR and high sample validity. Raw ASR alone can overstate practical risk when generated flows violate network constraints.</div>', unsafe_allow_html=True)

with baseline_tab:
    st.subheader("Clean-Test Performance")
    st.caption("Evaluate imbalanced NIDS data with F1, recall, and ROC-AUC—not accuracy alone.")
    if clean_df.empty:
        st.info("No clean model records were found.")
    else:
        long_clean = clean_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall / TPR", "F1-Score", "ROC-AUC"], var_name="Metric", value_name="Score")
        fig = px.bar(long_clean, x="Model", y="Score", color="Metric", barmode="group", text_auto=".1%")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
        st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displayModeBar": False})
        styled_table(clean_df, ["Accuracy", "Precision", "Recall / TPR", "F1-Score", "ROC-AUC"])
        st.subheader("Confusion Matrices")
        matrix_columns = st.columns(min(4, len(clean_df)))
        for column, (_, row) in zip(matrix_columns, clean_df.iterrows()):
            with column:
                if all(pd.notna(row[item]) for item in ["TN", "FP", "FN", "TP"]):
                    z = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
                    fig = px.imshow(z, text_auto=True, x=["Predicted Benign", "Predicted Attack"], y=["Actual Benign", "Actual Attack"], color_continuous_scale=[[0, "#0f1b2d"], [1, "#22d3ee"]], title=row["Model"])
                    fig.update_coloraxes(showscale=False)
                    st.plotly_chart(chart_layout(fig, 285), use_container_width=True, config={"displayModeBar": False})

with attack_tab:
    st.subheader("Constrained Evasion Analysis")
    st.caption("Valid ASR counts successful evasions after enforcing the project’s realistic feature constraints.")
    if attack_df.empty:
        st.info("No attack records were found.")
    else:
        filters, content = st.columns([1, 3])
        with filters:
            chosen_models = st.multiselect("Models", sorted(attack_df["Model"].unique()), default=sorted(attack_df["Model"].unique()))
            chosen_constraints = st.multiselect("Constraint Types", sorted(attack_df["Constraint"].unique()), default=sorted(attack_df["Constraint"].unique()))
        filtered = attack_df[attack_df["Model"].isin(chosen_models) & attack_df["Constraint"].isin(chosen_constraints)]
        with content:
            if filtered.empty:
                st.warning("No attack scenarios match the selected filters.")
            else:
                plot = filtered.melt(id_vars=["Model", "Attack", "Constraint"], value_vars=["Raw ASR", "Valid ASR"], var_name="Measure", value_name="ASR")
                plot["Scenario"] = plot["Model"] + " · " + plot["Attack"]
                fig = px.bar(plot, x="ASR", y="Scenario", color="Measure", barmode="group", orientation="h", text_auto=".1%", color_discrete_map={"Raw ASR": "#64748B", "Valid ASR": "#F97316"})
                fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Attack Success Rate")
                fig.update_layout(height=max(390, 54 * len(filtered)))
                st.plotly_chart(chart_layout(fig, max(390, 54 * len(filtered))), use_container_width=True, config={"displayModeBar": False})
        display_attack = filtered.copy()
        styled_table(display_attack, ["Validity Rate", "Raw ASR", "Valid ASR"], height=420)

with defense_tab:
    st.subheader("Defense Recovery & Robustness")
    st.caption("DRR measures recovered attack-induced loss; robustness normalizes attacked accuracy by clean accuracy.")
    if defense_df.empty:
        st.info("No defense records were found.")
    else:
        plot = defense_df.melt(id_vars="Model", value_vars=["Clean Accuracy", "Undefended Accuracy", "Defended Accuracy"], var_name="Condition", value_name="Accuracy")
        fig = px.bar(plot, x="Model", y="Accuracy", color="Condition", barmode="group", text_auto=".1%", color_discrete_map={"Clean Accuracy": "#60A5FA", "Undefended Accuracy": "#FB7185", "Defended Accuracy": "#2DD4BF"})
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
        st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displayModeBar": False})
        styled_table(defense_df, ["Clean Accuracy", "Undefended Accuracy", "Defended Accuracy", "Robustness Before", "Robustness After", "DRR", "Clean Cost"])

with matrix_tab:
    st.subheader("Decision Matrix")
    st.caption("A defense-ready view combining detection quality, adversarial exposure, and recovery.")
    if clean_df.empty:
        st.info("Clean metrics are required to build the decision matrix.")
    else:
        matrix = clean_df[["Model", "Accuracy", "Precision", "Recall / TPR", "F1-Score", "ROC-AUC"]].copy()
        if not attack_df.empty:
            attack_rollup = attack_df.groupby("Model", as_index=False).agg(**{"Mean Valid ASR": ("Valid ASR", "mean"), "Worst Valid ASR": ("Valid ASR", "max"), "Mean Validity": ("Validity Rate", "mean")})
            matrix = matrix.merge(attack_rollup, on="Model", how="left")
        if not defense_df.empty:
            defense_rollup = defense_df.groupby("Model", as_index=False).agg(**{"Defense DRR": ("DRR", "mean"), "Defended Robustness": ("Robustness After", "mean"), "Clean Cost": ("Clean Cost", "mean")})
            matrix = matrix.merge(defense_rollup, on="Model", how="left")
        exposure = matrix["Worst Valid ASR"].fillna(0) if "Worst Valid ASR" in matrix else pd.Series(0.0, index=matrix.index)
        matrix["Risk-Adjusted Score"] = matrix["F1-Score"] * (1 - exposure)
        matrix = matrix.sort_values(["Risk-Adjusted Score", "F1-Score"], ascending=False).reset_index(drop=True)
        matrix.insert(0, "Rank", range(1, len(matrix) + 1))
        st.markdown('<div class="note"><b>Ranking rule:</b> F1 × (1 − worst valid ASR). This transparent dashboard score supports comparison; it is not a replacement for the project’s official ASR, robustness, or DRR metrics.</div>', unsafe_allow_html=True)
        styled_table(matrix, [column for column in matrix.columns if column not in ["Rank", "Model"]])
        st.download_button("Download Evaluation Matrix", matrix.to_csv(index=False).encode("utf-8"), file_name=f"{dataset_name.lower()}_evaluation_matrix.csv", mime="text/csv", type="primary")

with operations_tab:
    st.subheader("Network Intrusion Monitoring")
    st.caption("Operational traffic visualization is separate from the research evaluation metrics above.")
    source_column, settings_column = st.columns([2, 1])
    with source_column:
        traffic_source = st.radio(
            "Traffic Source", ["Demo Traffic", "Upload Network Log"], horizontal=True,
            help="Demo traffic is synthetic and intended only to demonstrate the monitoring interface.",
        )
    traffic_df: pd.DataFrame | None = None
    if traffic_source == "Demo Traffic":
        with settings_column:
            demo_size = st.select_slider("Demo Events", options=[1000, 2500, 5000, 7500], value=2500)
            demo_seed = st.number_input("Demo Seed", min_value=0, max_value=9999, value=42)
        traffic_df = generate_demo_traffic(int(demo_size), int(demo_seed))
        st.warning("Demo mode: all events in this tab are synthetic and must not be presented as experiment results.", icon="⚠️")
    else:
        uploaded_log = st.file_uploader("Upload a CSV network or flow log", type=["csv"], help="Common timestamp, IP, port, protocol, label, severity, action, and country column names are detected automatically.")
        if uploaded_log is not None:
            try:
                traffic_df = normalize_traffic_csv(pd.read_csv(uploaded_log, low_memory=False))
            except Exception as exc:
                st.error(f"Could not read this network log: {exc}")
        else:
            st.info("Upload a CSV file to open the operational monitoring view.")

    if traffic_df is not None and not traffic_df.empty:
        benign_names = {"benign", "normal", "0", "allowed"}
        traffic_df = traffic_df.copy()
        traffic_df["is_malicious"] = ~traffic_df["label"].str.lower().isin(benign_names)
        traffic_df["is_blocked"] = traffic_df["action"].str.lower().str.contains("block", na=False)
        start_time, end_time = traffic_df["timestamp"].min(), traffic_df["timestamp"].max()
        duration_hours = max((end_time - start_time).total_seconds() / 3600, 1 / 3600)
        total_events = len(traffic_df)
        malicious_events = int(traffic_df["is_malicious"].sum())
        blocked_events = int(traffic_df["is_blocked"].sum())
        critical_events = int((traffic_df["severity"].str.lower() == "critical").sum())
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Events", f"{total_events:,}", f"{duration_hours:.1f} h window")
        m2.metric("Malicious", f"{malicious_events:,}", percent(malicious_events / total_events), delta_color="inverse")
        m3.metric("Blocked", f"{blocked_events:,}", percent(blocked_events / max(malicious_events, 1)))
        m4.metric("Critical", f"{critical_events:,}", "Priority incidents", delta_color="inverse")
        m5.metric("Source IPs", f"{traffic_df['src_ip'].nunique():,}", "Unique observed")
        m6.metric("Event Rate", f"{total_events / (duration_hours * 3600):.2f}", "Events / second")

        timeline_column, map_column = st.columns([1.35, 1])
        with timeline_column:
            st.markdown("#### Event Activity")
            interval = "15min" if duration_hours <= 48 else "1h"
            timeline = traffic_df.set_index("timestamp").resample(interval).agg(Total=("label", "size"), Malicious=("is_malicious", "sum"), Blocked=("is_blocked", "sum")).reset_index()
            timeline_long = timeline.melt(id_vars="timestamp", var_name="Event Type", value_name="Events")
            fig = px.area(timeline_long, x="timestamp", y="Events", color="Event Type", color_discrete_map={"Total": "#60A5FA", "Malicious": "#FB7185", "Blocked": "#2DD4BF"})
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(chart_layout(fig, 380), use_container_width=True, config={"displayModeBar": False})
        with map_column:
            st.markdown("#### Malicious Traffic Origins")
            country_counts = traffic_df[traffic_df["is_malicious"] & traffic_df["country"].ne("Unknown")].groupby("country").size().reset_index(name="Events")
            if country_counts.empty:
                st.info("Country data is unavailable in this traffic source.")
            else:
                fig = px.choropleth(country_counts, locations="country", locationmode="country names", color="Events", color_continuous_scale=[[0, "#172033"], [.5, "#F59E0B"], [1, "#EF4444"]])
                fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor="#172033", countrycolor="#334155", showframe=False)
                fig.update_layout(coloraxis_colorbar=dict(title="Events", thickness=12))
                st.plotly_chart(chart_layout(fig, 380), use_container_width=True, config={"displayModeBar": False})

        ip_column, port_column, protocol_column = st.columns(3)
        with ip_column:
            st.markdown("#### Top Malicious Source IPs")
            top_ips = traffic_df[traffic_df["is_malicious"]].groupby("src_ip").size().nlargest(10).sort_values().reset_index(name="Events")
            if top_ips.empty:
                st.info("No malicious source IPs detected.")
            else:
                fig = px.bar(top_ips, x="Events", y="src_ip", orientation="h", text="Events", color="Events", color_continuous_scale="Reds")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(chart_layout(fig, 390), use_container_width=True, config={"displayModeBar": False})
        with port_column:
            st.markdown("#### Top Destination Ports")
            top_ports = traffic_df[traffic_df["dst_port"] > 0].groupby("dst_port").size().nlargest(10).sort_values().reset_index(name="Events")
            top_ports["Port / Service"] = top_ports["dst_port"].map(port_label)
            if top_ports.empty:
                st.info("Destination-port data is unavailable.")
            else:
                fig = px.bar(top_ports, x="Events", y="Port / Service", orientation="h", text="Events", color="Events", color_continuous_scale="Blues")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(chart_layout(fig, 390), use_container_width=True, config={"displayModeBar": False})
        with protocol_column:
            st.markdown("#### Protocol Distribution")
            protocols = traffic_df["protocol"].value_counts().head(10).reset_index()
            protocols.columns = ["Protocol", "Events"]
            fig = px.pie(protocols, names="Protocol", values="Events", hole=.58, color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(chart_layout(fig, 390), use_container_width=True, config={"displayModeBar": False})

        attack_column, severity_column, action_column = st.columns(3)
        with attack_column:
            st.markdown("#### Attack Categories")
            attacks = traffic_df[traffic_df["is_malicious"]]["label"].value_counts().head(8).sort_values().reset_index()
            attacks.columns = ["Attack", "Events"]
            if attacks.empty:
                st.info("No attack labels detected.")
            else:
                fig = px.bar(attacks, x="Events", y="Attack", orientation="h", text="Events", color="Events", color_continuous_scale="OrRd")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(chart_layout(fig, 340), use_container_width=True, config={"displayModeBar": False})
        with severity_column:
            st.markdown("#### Severity Profile")
            severity_order = ["Critical", "High", "Medium", "Low", "Info"]
            severity = traffic_df["severity"].value_counts().reindex(severity_order).fillna(0).reset_index()
            severity.columns = ["Severity", "Events"]
            fig = px.pie(severity, names="Severity", values="Events", hole=.58, color="Severity", color_discrete_map={"Critical": "#EF4444", "High": "#F97316", "Medium": "#FBBF24", "Low": "#2DD4BF", "Info": "#64748B"})
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True, config={"displayModeBar": False})
        with action_column:
            st.markdown("#### Response Actions")
            actions = traffic_df["action"].value_counts().reset_index()
            actions.columns = ["Action", "Events"]
            fig = px.pie(actions, names="Action", values="Events", hole=.58, color_discrete_sequence=["#2DD4BF", "#FBBF24", "#FB7185", "#60A5FA"])
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Recent High-Priority Events")
        priority = traffic_df[traffic_df["severity"].str.lower().isin(["critical", "high"])].head(50)
        event_columns = ["timestamp", "severity", "label", "action", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packet_length", "country"]
        st.dataframe(priority[event_columns], hide_index=True, use_container_width=True, height=430)
        st.caption(f"Showing {len(priority)} of {total_events:,} events · newest first")

with st.expander("Experiment Provenance & Limitations"):
    st.markdown(
        f"- Source artifact: `{selected.relative_to(ROOT)}`\n"
        f"- Dataset identifier: `{summary.get('dataset', selected.parent.name)}`\n"
        f"- Reproducibility seed: `{summary.get('seed', 'not recorded')}`\n"
        "- Valid ASR should be interpreted together with validity rate and attack sample size.\n"
        "- Dashboard rankings are descriptive summaries of recorded runs, not statistical significance tests.\n"
        "- Synthetic or manually prepared summaries must not be presented as real CICIDS2017 or UNSW-NB15 experiment results."
    )

st.divider()
st.caption("Aegis · Adversarial NIDS Evaluation Console · Offline experiment replay")
