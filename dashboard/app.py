"""
Professional NIDS / SIEM Dashboard — True Senior Refactor
========================================================
- Research mode (real summary.json from pipeline)
- Upload CSV (now actually extracts ports)
- Demo Synthetic Logs (SOC-grade)
Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODELS = ["rf", "xgb", "svm", "mlp"]
MODEL_LABELS = {
    "rf": "Random Forest", "xgb": "XGBoost",
    "svm": "SVM", "mlp": "MLP (Neural Net)",
}
MODEL_COLORS = {
    "rf": "#22c55e", "xgb": "#3b82f6",
    "svm": "#f59e0b", "mlp": "#a855f7",
}

# Common ports → human readable (this is the visual fix you asked for)
PORT_NAMES = {
    20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS",
    445: "SMB", 993: "IMAPS", 995: "POP3S", 1433: "MSSQL", 3306: "MySQL",
    3389: "RDP", 5432: "PostgreSQL", 5900: "VNC", 8080: "HTTP-Alt",
    8443: "HTTPS-Alt",
}

ATTACK_COUNTRIES = [
    ("China", "CN", 32.0), ("United States", "US", 16.0), ("Russia", "RU", 13.0),
    ("India", "IN", 9.0), ("Brazil", "BR", 6.5), ("Germany", "DE", 5.0),
    ("Netherlands", "NL", 4.5), ("France", "FR", 3.5), ("United Kingdom", "GB", 3.0),
    ("South Korea", "KR", 2.5), ("Iran", "IR", 2.5), ("Vietnam", "VN", 2.0),
]
PROTOCOLS = ["TCP", "UDP", "HTTP", "HTTPS", "ICMP", "DNS", "SSH", "FTP", "SMTP", "RDP"]
ATTACK_TYPES = [
    "Benign", "DoS", "DDoS", "PortScan", "BruteForce",
    "Web Attack", "Bot", "Infiltration", "Heartbleed", "SQL Injection"
]
SEVERITIES = ["Critical", "High", "Medium", "Low", "Info"]
POPULAR_DST_PORTS = [80, 443, 22, 53, 3389, 8080, 21, 25, 445, 1433, 3306, 5900, 8443, 23, 110]

# ──────────────────────────────────────────────
# PAGE + THEME
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS Adversarial Robustness Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background: #0b0f19; color: #e2e8f0; }
    .block-container { padding-top: 1.1rem; max-width: 1480px; }
    section[data-testid="stSidebar"] {
        background: #0f172a; border-right: 1px solid #1e293b;
    }
    .kpi-card {
        background: linear-gradient(145deg, #111827, #1e293b);
        border: 1px solid #1e293b; border-radius: 14px;
        padding: 1.05rem 1.2rem; height: 100%;
    }
    .kpi-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { font-size: 1.65rem; font-weight: 700; color: #f8fafc; margin-top: 0.15rem; }
    .kpi-sub { font-size: 0.7rem; color: #64748b; margin-top: 0.2rem; }
    .section-title {
        font-size: 1.1rem; font-weight: 650; color: #f1f5f9;
        margin: 1.5rem 0 0.7rem 0; padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e293b;
    }
    .insight-box {
        background: #0f172a; border-left: 4px solid #3b82f6;
        border-radius: 0 10px 10px 0; padding: 0.85rem 1.1rem;
        margin: 0.7rem 0; font-size: 0.87rem; color: #cbd5e1;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b; border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.15rem; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important; color: #f8fafc !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def safe_get(d: Any, *keys, default=0):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default

def pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except Exception:
        return "—"

def apply_dark_layout(fig: go.Figure, height: int = 320, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        margin=dict(t=30, b=20),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        showlegend=showlegend,
    )
    return fig

def render_kpi(label: str, value: str, sub: str = "", color: str = "#f8fafc") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """

def port_label(port) -> str:
    """Make destination ports actually readable — this is the visual fix."""
    try:
        p = int(port)
        name = PORT_NAMES.get(p)
        return f"{p} ({name})" if name else str(p)
    except Exception:
        return str(port)

# ──────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_summary(dataset: str) -> Optional[Dict]:
    path = ROOT / "results" / dataset / "summary.json"
    if not path.exists():
        # also try official naming
        path = ROOT / "results" / f"{dataset}2017" / "summary.json" if dataset == "cicids" else path
        if not path.exists():
            return None
    with open(path) as f:
        return json.load(f)

@st.cache_data(show_spinner="Generating synthetic traffic…")
def generate_synthetic_logs(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    popular_src = [
        f"{rng.integers(1, 200)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
        for _ in range(45)
    ]
    now = datetime.now()
    rows = []
    for i in range(n_rows):
        if i % 35 < 9:
            attack = random.choices(
                ["DoS", "DDoS", "PortScan", "BruteForce", "Bot"],
                weights=[28, 22, 22, 15, 13]
            )[0]
            severity = random.choices(["Critical", "High", "Medium"], weights=[28, 48, 24])[0]
        else:
            attack = random.choices(ATTACK_TYPES, weights=[60, 10, 8, 7, 5, 4, 3, 2, 0.5, 0.5])[0]
            severity = random.choices(SEVERITIES, weights=[6, 15, 34, 29, 16])[0]

        country, iso, _ = random.choices(ATTACK_COUNTRIES, weights=[c[2] for c in ATTACK_COUNTRIES])[0]
        protocol = random.choice(PROTOCOLS)
        src_ip = random.choice(popular_src) if random.random() < 0.68 else (
            f"{rng.integers(1, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
        )
        dst_ip = f"10.{rng.integers(0, 18)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
        dst_port = random.choice(POPULAR_DST_PORTS) if random.random() < 0.78 else int(rng.integers(1024, 65535))
        pkt_len = max(40, min(int(rng.lognormal(6.15, 0.88)), 1500))

        if attack == "Benign":
            action = "allow"
            severity = random.choice(["Info", "Low", "Medium"])
        else:
            action = random.choices(["block", "alert", "allow"], weights=[55, 30, 15])[0]

        ts = now - timedelta(minutes=int(rng.integers(0, 60 * 36)))
        rows.append({
            "timestamp": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": int(rng.integers(1024, 65535)),
            "dst_port": dst_port,
            "protocol": protocol,
            "packet_length": pkt_len,
            "label": attack,
            "severity": severity,
            "action": action,
            "country": country,
            "country_iso": iso,
            "rule": f"SID-{rng.integers(100000, 999999)}",
        })
    return (
        pd.DataFrame(rows)
        .sort_values("timestamp", ascending=False)
        .reset_index(drop=True)
    )

def normalize_uploaded_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Now actually extracts ports. No more fake zeros."""
    lower = {c.lower().strip(): c for c in df.columns}

    def find(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    ts_col = find("timestamp", "time", "datetime", "date", "ts")
    src_col = find("src_ip", "source_ip", "srcip", "source ip", "src")
    dst_col = find("dst_ip", "destination_ip", "dstip", "destination ip", "dst")
    proto_col = find("protocol", "proto")
    len_col = find("packet_length", "pkt_len", "length", "bytes")
    label_col = find("label", "attack", "class", "category", "attack_cat")
    sev_col = find("severity", "priority", "risk")
    action_col = find("action", "verdict")
    country_col = find("country", "src_country")
    src_port_col = find("src_port", "source_port", "sport")
    dst_port_col = find("dst_port", "destination_port", "dport", "port")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.Timestamp.now()
    out["src_ip"] = df[src_col].astype(str) if src_col else "0.0.0.0"
    out["dst_ip"] = df[dst_col].astype(str) if dst_col else "0.0.0.0"
    out["protocol"] = df[proto_col].astype(str).str.upper() if proto_col else "TCP"
    out["packet_length"] = pd.to_numeric(df[len_col], errors="coerce").fillna(64) if len_col else 64
    out["label"] = df[label_col].astype(str) if label_col else "Unknown"
    out["severity"] = df[sev_col].astype(str) if sev_col else "Medium"
    out["action"] = df[action_col].astype(str).str.lower() if action_col else "allow"
    out["country"] = df[country_col].astype(str) if country_col else "Unknown"
    out["rule"] = "CUSTOM"

    # FIXED: real ports now
    out["src_port"] = pd.to_numeric(df[src_port_col], errors="coerce").fillna(0).astype(int) if src_port_col else 0
    out["dst_port"] = pd.to_numeric(df[dst_port_col], errors="coerce").fillna(0).astype(int) if dst_port_col else 0

    sev_map = {
        "1": "Critical", "2": "High", "3": "Medium", "4": "Low", "5": "Info",
        "critical": "Critical", "high": "High", "medium": "Medium",
        "low": "Low", "info": "Info",
    }
    out["severity"] = out["severity"].str.lower().map(lambda x: sev_map.get(x, str(x).title()))
    return out.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ NIDS Dashboard")
    st.caption("Adversarial Robustness + SOC View")
    st.divider()

    data_mode = st.radio(
        "Data Source",
        ["Research (summary.json)", "Upload Network Logs (CSV)", "Demo Synthetic Logs"],
        index=2,
    )

    summary = None
    logs_df = None
    dataset = "cicids"

    if data_mode == "Research (summary.json)":
        dataset = st.selectbox(
            "Dataset",
            ["cicids", "unsw"],
            format_func=lambda x: "CICIDS2017" if x == "cicids" else "UNSW-NB15",
        )
        summary = load_summary(dataset)
        if summary is None:
            st.error("No summary.json found")
            st.info("Expected: `results/cicids/summary.json` or `results/unsw/summary.json`")
            st.caption("Run the pipeline first: `python scripts/run_all.py`")
    elif data_mode == "Upload Network Logs (CSV)":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded, low_memory=False)
            logs_df = normalize_uploaded_csv(raw)
            st.success(f"Loaded {len(logs_df):,} events")
        else:
            st.info("Upload any CSV with network features (flexible column names)")
    else:
        n_rows = st.slider("Number of synthetic events", 1500, 7000, 4000, 500)
        seed = st.number_input("Seed (reproducible)", 0, 9999, 42)
        logs_df = generate_synthetic_logs(n_rows, seed=seed)
        st.success(f"Generated {len(logs_df):,} realistic events (seed={seed})")

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Research mode → adversarial robustness results\n"
        "Demo mode → SOC operational view for presentation."
    )

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
  <div style="font-size:2.1rem">🛡️</div>
  <div>
    <h1 style="margin:0;font-size:1.5rem;font-weight:700;color:#f8fafc">
      Adversarial Robustness of ML-Based NIDS
    </h1>
    <p style="margin:0.12rem 0 0;color:#94a3b8;font-size:0.86rem">
      Research Evaluation Framework + Professional SOC Demo
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════
# RESEARCH VIEW
# ══════════════════════════════════════════════
if summary is not None:
    # Handle both your old keys and the real pipeline keys
    clean_metrics = summary.get("clean_metrics") or summary.get("clean", {})
    attack_results = summary.get("attack_results", [])
    defense_summary = summary.get("defense_summary", {})
    robustness_table = summary.get("robustness_table", [])

    # Convert official nested format → flat list
    if not attack_results and "attacks_before" in summary:
        tmp = []
        for model, attacks in summary["attacks_before"].items():
            for atk, rec in attacks.items():
                tmp.append({
                    "model": model,
                    "attack": atk,
                    "valid_attack_success": rec.get("asr", 0),
                    "asr": rec.get("asr", 0),
                })
        attack_results = tmp

    st.markdown("## 🔬 Research Evaluation — Core Contribution")

    best_acc = max(
        (safe_get(clean_metrics, m, "accuracy") for m in MODELS if m in clean_metrics),
        default=0,
    )
    asr_vals = [r.get("valid_attack_success") or r.get("asr") or 0 for r in attack_results]
    worst_asr = max(asr_vals) if asr_vals else 0
    drr_vals = [r.get("defense_recovery_rate") or 0 for r in robustness_table]
    best_drr = max(drr_vals) if drr_vals else 0
    cost_dict = defense_summary.get("adversarial_training_clean_cost", {})
    avg_cost = sum(cost_dict.values()) / len(cost_dict) if cost_dict else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(render_kpi("Best Clean Accuracy", pct(best_acc), "Before attacks"), unsafe_allow_html=True)
    c2.markdown(render_kpi("Worst Valid ASR", pct(worst_asr), "Under constraints", "#ef4444"), unsafe_allow_html=True)
    c3.markdown(render_kpi("Best Defense Recovery", pct(best_drr), "After defense", "#22c55e"), unsafe_allow_html=True)
    c4.markdown(render_kpi("Avg Accuracy Cost", pct(avg_cost), "Defense overhead", "#a855f7"), unsafe_allow_html=True)
    
    #st.markdown("""
    #<div class="insight-box">
    #    <b>Core Scientific Contribution</b><br>
    #    Most previous papers evaluate adversarial attacks on NIDS without realistic constraints.
    #    This work forces the attacker to respect real network feature limitations and systematically
    #    compares <b>Naive</b> (unrealistic) versus <b>Consistent</b> (more realistic) attacks.
    #</div>
    #""", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Clean Performance", "Attack Analysis (Naive vs Consistent)", "Defense Recovery"
    ])

    with tab1:
        if clean_metrics:
            acc_df = pd.DataFrame([
                {
                    "Model": MODEL_LABELS.get(m, m),
                    "Accuracy": safe_get(clean_metrics, m, "accuracy"),
                    "key": m,
                }
                for m in MODELS if m in clean_metrics
            ])
            if not acc_df.empty:
                fig = px.bar(
                    acc_df, x="Model", y="Accuracy", color="key",
                    color_discrete_map=MODEL_COLORS,
                    text=[f"{v:.1%}" for v in acc_df["Accuracy"]],
                )
                fig.update_traces(textposition="outside")
                fig = apply_dark_layout(fig, height=340, showlegend=False)
                fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if clean_metrics:
            rows = []
            for m in MODELS:
                if m not in clean_metrics:
                    continue
                cm = clean_metrics[m]
                rows.append({
                    "Model": MODEL_LABELS.get(m, m),
                    "Accuracy": f"{safe_get(cm, 'accuracy'):.4f}",
                    "Precision": f"{safe_get(cm, 'precision'):.4f}",
                    "Recall": f"{safe_get(cm, 'recall'):.4f}",
                    "F1": f"{safe_get(cm, 'f1'):.4f}",
                    "ROC-AUC": f"{safe_get(cm, 'roc_auc'):.4f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    with tab3:
        st.markdown("### ⚔️ Attack Results — Naive vs Consistent")
        if not attack_results:
            st.warning("No attack results found in summary.json")
        else:
            df_atk = pd.DataFrame(attack_results)
            st.dataframe(df_atk, use_container_width=True, height=250)

            naive = df_atk[df_atk["attack"].astype(str).str.contains("naive", case=False, na=False)]
            consistent = df_atk[df_atk["attack"].astype(str).str.contains("consistent", case=False, na=False)]

            col_n, col_c = st.columns(2)
            with col_n:
                st.markdown("**🔴 Naive Attacks** (unrealistic)")
                if not naive.empty:
                    asr_col = "valid_attack_success" if "valid_attack_success" in naive else "asr"
                    st.metric("Avg Valid ASR", pct(naive[asr_col].mean()))
                    st.dataframe(naive, use_container_width=True)
                else:
                    st.caption("No naive data")
            with col_c:
                st.markdown("**🟢 Consistent Attacks** (realistic)")
                if not consistent.empty:
                    asr_col = "valid_attack_success" if "valid_attack_success" in consistent else "asr"
                    st.metric("Avg Valid ASR", pct(consistent[asr_col].mean()))
                    st.dataframe(consistent, use_container_width=True)
                else:
                    st.caption("No consistent data")

            if not naive.empty and not consistent.empty:
                asr_col = "valid_attack_success" if "valid_attack_success" in df_atk.columns else "asr"
                cmp = pd.DataFrame({
                    "Type": ["Naive", "Consistent"],
                    "Avg Valid ASR": [
                        naive[asr_col].mean(),
                        consistent[asr_col].mean(),
                    ],
                })
                fig = px.bar(
                    cmp, x="Type", y="Avg Valid ASR", color="Type",
                    color_discrete_map={"Naive": "#ef4444", "Consistent": "#22c55e"},
                    text=[f"{v:.1%}" for v in cmp["Avg Valid ASR"]],
                )
                fig.update_traces(textposition="outside")
                fig = apply_dark_layout(fig, height=320, showlegend=False)
                fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
                st.plotly_chart(fig, use_container_width=True)

                # st.markdown("""
                # <div class="insight-box">
                #     <b>Interpretation</b><br>
                #     If Consistent ASR is clearly lower than Naive ASR, previous works overestimated
                #     real-world attack success by ignoring feature constraints. This is the central finding.
                # </div>
                # """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 🛡️ Defense Recovery")
        if robustness_table:
            df_rob = pd.DataFrame(robustness_table)
            st.dataframe(df_rob, use_container_width=True)
            if "defense_recovery_rate" in df_rob.columns:
                drr = df_rob[["model", "defense_recovery_rate"]].dropna()
                drr["model"] = drr["model"].map(MODEL_LABELS).fillna(drr["model"])
                fig = px.bar(
                    drr, x="model", y="defense_recovery_rate", color="model",
                    color_discrete_map={v: MODEL_COLORS.get(k) for k, v in MODEL_LABELS.items()},
                    text=[f"{v:.1%}" for v in drr["defense_recovery_rate"]],
                )
                fig.update_traces(textposition="outside")
                fig = apply_dark_layout(fig, height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No defense recovery data available in this summary.json")

# ══════════════════════════════════════════════
# OPERATIONAL / DEMO VIEW
# ══════════════════════════════════════════════
elif logs_df is not None and len(logs_df) > 0:
    st.markdown("## 🖥️ Security Operations Center — Live Demo View")

    logs_df = logs_df.copy()
    logs_df["is_malicious"] = logs_df["label"].str.lower() != "benign"
    logs_df["is_blocked"] = logs_df["action"].str.contains("block", case=False, na=False)

    total_events = len(logs_df)
    critical = (logs_df["severity"] == "Critical").sum()
    high = (logs_df["severity"] == "High").sum()
    blocked = logs_df["is_blocked"].sum()
    malicious = logs_df["is_malicious"].sum()
    unique_src = logs_df["src_ip"].nunique()
    time_span_hours = max(
        (logs_df["timestamp"].max() - logs_df["timestamp"].min()).total_seconds() / 3600, 0.1
    )
    avg_eps = total_events / (time_span_hours * 3600)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(render_kpi("Total Events", f"{total_events:,}", f"Last {time_span_hours:.1f}h"), unsafe_allow_html=True)
    k2.markdown(render_kpi("Critical", str(critical), "Immediate", "#ef4444"), unsafe_allow_html=True)
    k3.markdown(render_kpi("High Severity", str(high), "Elevated", "#f97316"), unsafe_allow_html=True)
    k4.markdown(render_kpi("Blocked", f"{blocked:,}", f"{blocked/total_events*100:.1f}%", "#eab308"), unsafe_allow_html=True)
    k5.markdown(render_kpi("Malicious", f"{malicious:,}", f"{malicious/total_events*100:.1f}%", "#f43f5e"), unsafe_allow_html=True)
    k6.markdown(render_kpi("Avg EPS", f"{avg_eps:.1f}", "Events/sec", "#3b82f6"), unsafe_allow_html=True)

    st.markdown("")

    col_time, col_map = st.columns([1.55, 1])
    with col_time:
        st.markdown('<div class="section-title">📈 Event Volume Over Time</div>', unsafe_allow_html=True)
        ts = (
            logs_df.set_index("timestamp")
            .resample("15min")
            .agg({"label": "count", "is_malicious": "sum", "is_blocked": "sum"})
            .reset_index()
        )
        ts.columns = ["timestamp", "Total", "Malicious", "Blocked"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["timestamp"], y=ts["Total"], name="Total",
            fill="tozeroy", line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=ts["timestamp"], y=ts["Malicious"], name="Malicious",
            fill="tozeroy", line=dict(color="#ef4444", width=2),
            fillcolor="rgba(239,68,68,0.12)",
        ))
        fig.add_trace(go.Scatter(
            x=ts["timestamp"], y=ts["Blocked"], name="Blocked",
            line=dict(color="#eab308", width=2, dash="dot"),
        ))
        fig = apply_dark_layout(fig, height=320)
        fig.update_layout(legend=dict(orientation="h", y=1.12), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with col_map:
        st.markdown('<div class="section-title">🌍 Attack Origin Hotspots</div>', unsafe_allow_html=True)
        country_counts = (
            logs_df[logs_df["is_malicious"]]
            .groupby("country")
            .size()
            .reset_index(name="count")
        )
        if not country_counts.empty:
            fig = px.choropleth(
                country_counts,
                locations="country",
                locationmode="country names",
                color="count",
                color_continuous_scale="YlOrRd",
            )
            fig.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                geo=dict(
                    bgcolor="rgba(0,0,0,0)",
                    landcolor="#1e293b",
                    showcountries=True,
                    countrycolor="#334155",
                    projection_type="natural earth",
                ),
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar=dict(title="Attacks", thickness=14),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── FIXED Top Destination Ports ───────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">🔝 Top Malicious Source IPs</div>', unsafe_allow_html=True)
        top_src = (
            logs_df[logs_df["is_malicious"]]
            .groupby("src_ip")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_src, x="count", y="src_ip", orientation="h",
            text="count", color="count", color_continuous_scale="Reds",
        )
        fig.update_traces(textposition="outside", textfont_size=12, textfont_color="#e2e8f0")
        fig = apply_dark_layout(fig, height=360, showlegend=False)
        fig.update_layout(
            yaxis=dict(autorange="reversed", title="", tickfont=dict(size=11)),
            xaxis=dict(title="Event Count"),
            margin=dict(t=10, b=30, l=10, r=50),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">🎯 Top Destination Ports</div>', unsafe_allow_html=True)

        # REAL FIX: map ports to human names + keep only real ports
        top_ports = (
            logs_df[logs_df["dst_port"] > 0]          # kill the zero garbage
            .groupby("dst_port")
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(10)
        )
        top_ports["Port"] = top_ports["dst_port"].apply(port_label)

        fig = px.bar(
            top_ports,
            x="Count",
            y="Port",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Blues",
        )
        fig.update_traces(textposition="outside")
        fig = apply_dark_layout(fig, height=360, showlegend=False)
        fig.update_layout(
            yaxis=dict(autorange="reversed", title="", tickfont=dict(size=12)),
            xaxis=dict(title="Count"),
            margin=dict(t=10, b=30, l=10, r=60),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">⚠️ Severity Distribution</div>', unsafe_allow_html=True)
        sev_order = ["Critical", "High", "Medium", "Low", "Info"]
        sev_counts = logs_df["severity"].value_counts().reindex(sev_order).fillna(0).reset_index()
        sev_counts.columns = ["Severity", "Count"]
        colors = {
            "Critical": "#ef4444", "High": "#f97316",
            "Medium": "#eab308", "Low": "#22c55e", "Info": "#64748b",
        }
        fig = px.pie(
            sev_counts, names="Severity", values="Count", hole=0.55,
            color="Severity", color_discrete_map=colors,
        )
        fig.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            margin=dict(t=20, b=20),
            legend=dict(orientation="v", y=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    # rest of the charts stay the same (Attack Categories, Protocol, Action, Packet Length, High-sev table)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">🔥 Top Attack Categories</div>', unsafe_allow_html=True)
        top_atk = (
            logs_df[logs_df["is_malicious"]]["label"]
            .value_counts()
            .head(8)
            .reset_index()
        )
        top_atk.columns = ["Attack", "Count"]
        fig = px.bar(
            top_atk, x="Count", y="Attack", orientation="h",
            text="Count", color="Count", color_continuous_scale="OrRd",
        )
        fig.update_traces(textposition="outside")
        fig = apply_dark_layout(fig, height=300, showlegend=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=""), margin=dict(t=10, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">📡 Protocol Distribution</div>', unsafe_allow_html=True)
        proto = logs_df["protocol"].value_counts().reset_index()
        proto.columns = ["Protocol", "Count"]
        fig = px.pie(
            proto, names="Protocol", values="Count", hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">🛡️ Action Breakdown</div>', unsafe_allow_html=True)
        act = logs_df["action"].value_counts().reset_index()
        act.columns = ["Action", "Count"]
        act_colors = {"block": "#ef4444", "alert": "#f59e0b", "allow": "#22c55e"}
        fig = px.pie(
            act, names="Action", values="Count", hole=0.5,
            color="Action", color_discrete_map=act_colors,
        )
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📦 Packet Length Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(logs_df, x="packet_length", nbins=50, color_discrete_sequence=["#6366f1"])
    fig = apply_dark_layout(fig, height=250)
    fig.update_layout(
        xaxis=dict(title="Packet Length (bytes)"),
        yaxis=dict(title="Count"),
        margin=dict(t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🚨 Recent High-Severity Events</div>', unsafe_allow_html=True)
    high_sev = logs_df[logs_df["severity"].isin(["Critical", "High"])].head(40)

    def color_sev(val):
        if val == "Critical":
            return "background-color: #7f1d1d; color: #fecaca"
        if val == "High":
            return "background-color: #7c2d12; color: #fed7aa"
        return ""

    cols = [
        "timestamp", "severity", "label", "action", "src_ip", "dst_ip",
        "dst_port", "protocol", "packet_length", "country", "rule",
    ]
    st.dataframe(
        high_sev[cols].style.map(color_sev, subset=["severity"]),
        use_container_width=True,
        height=380,
    )
    st.caption(f"Showing {len(high_sev)} high-severity events out of {total_events:,} total")

else:
    st.info("Select a data source in the sidebar to begin.")

st.markdown("---")
st.caption("NIDS Adversarial Robustness Dashboard - Demo")