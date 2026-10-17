import os
import time
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

APP_NAME = "SentinelAI"
st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #112a43 0%, #07111f 42%, #050b14 100%); color: #e7f0fa; }
section[data-testid="stSidebar"] { background: #081522; border-right: 1px solid #20364b; }
section[data-testid="stSidebar"] * { color: #d8e8f5; }
h1,h2,h3 { letter-spacing: -0.035em; }
div[data-testid="stMetric"] { background: linear-gradient(145deg, #12273a, #0b1928); border: 1px solid #23435b; padding: 17px 18px; border-radius: 15px; }
div[data-testid="stMetricLabel"] { color: #9db6cb; }
div[data-testid="stMetricValue"] { color: #f4f9ff; }
.stButton button, .stDownloadButton button { border-radius: 10px; border: 1px solid #2d7892; background: linear-gradient(120deg,#0e7490,#155e75); color: white; font-weight: 700; }
.stButton button:hover, .stDownloadButton button:hover { border-color: #67e8f9; color: white; }
div[data-testid="stDataFrame"] { border: 1px solid #203d54; border-radius: 12px; overflow: hidden; }
div[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stTabs"] button { font-weight: 700; }
.small-muted { color: #92abc0; font-size: .88rem; }
.eyebrow { color: #67e8f9; text-transform: uppercase; letter-spacing: .16em; font-size: .74rem; font-weight: 800; }
.hero { padding: 22px 26px; border: 1px solid #25465f; border-radius: 18px; background: linear-gradient(110deg, rgba(14,116,144,.22), rgba(16,31,49,.82)); margin-bottom: 18px; }
.pill { display: inline-block; padding: 5px 10px; border-radius: 999px; font-size: .75rem; font-weight: 800; background: #17354a; color: #8be9f5; border: 1px solid #2b6075; }
hr { border-color: #20384d; }
code { color: #8be9f5 !important; }
</style>
""", unsafe_allow_html=True)

SEVERITY_COLORS = {"Low": "#38bdf8", "Medium": "#facc15", "High": "#fb923c", "Critical": "#fb7185"}
ATTACK_LABELS = {"normal": "Normal", "benign": "Normal", "0": "Normal", "dos": "DoS", "ddos": "DDoS", "probe": "Probe / Scan",
                 "reconnaissance": "Reconnaissance", "exploits": "Exploit", "generic": "Generic Attack", "fuzzers": "Fuzzing",
                 "backdoor": "Backdoor", "analysis": "Analysis", "worms": "Worm", "shellcode": "Shellcode", "infiltration": "Infiltration",
                 "bot": "Bot / Botnet", "theft": "Theft", "password": "Credential Attack", "web attack": "Web Attack"}

def severity_from_risk(risk):
    if risk >= 85: return "Critical"
    if risk >= 65: return "High"
    if risk >= 35: return "Medium"
    return "Low"

def normalise_label(value):
    s = str(value).strip().lower()
    for key, label in ATTACK_LABELS.items():
        if key in s:
            return label
    return "Suspicious / Other" if s not in {"", "nan", "none"} else "Unknown"

def sample_events(n=240, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    now = pd.Timestamp.now().floor("s")
    ips = ["10.0.0.12", "10.0.0.24", "192.168.1.18", "172.16.0.9", "203.0.113.41", "198.51.100.23", "192.0.2.77"]
    kinds = ["Normal user activity", "Repeated failed logins", "Unusual request frequency",
             "Port scan pattern", "Large outbound transfer", "Suspicious API access", "Repeated 404 responses"]
    weights = [0.64, 0.10, 0.09, 0.05, 0.04, 0.05, 0.03]
    for i in range(n):
        kind = rng.choice(kinds, p=weights)
        if kind == "Normal user activity":
            risk = int(rng.integers(2, 28)); label = "Normal"; failed = int(rng.integers(0, 2)); rate = int(rng.integers(1, 25))
        elif kind == "Repeated failed logins":
            risk = int(rng.integers(62, 95)); label = "Credential Attack"; failed = int(rng.integers(5, 30)); rate = int(rng.integers(8, 80))
        elif kind == "Unusual request frequency":
            risk = int(rng.integers(38, 84)); label = "Potential Abuse"; failed = int(rng.integers(0, 5)); rate = int(rng.integers(50, 400))
        elif kind == "Port scan pattern":
            risk = int(rng.integers(70, 99)); label = "Reconnaissance"; failed = int(rng.integers(0, 4)); rate = int(rng.integers(30, 180))
        elif kind == "Large outbound transfer":
            risk = int(rng.integers(40, 90)); label = "Possible Exfiltration"; failed = int(rng.integers(0, 3)); rate = int(rng.integers(5, 60))
        elif kind == "Suspicious API access":
            risk = int(rng.integers(48, 96)); label = "Suspicious API Behaviour"; failed = int(rng.integers(0, 8)); rate = int(rng.integers(20, 220))
        else:
            risk = int(rng.integers(35, 80)); label = "Web Attack"; failed = int(rng.integers(0, 6)); rate = int(rng.integers(15, 120))
        rows.append({
            "timestamp": now - pd.Timedelta(seconds=int((n-i)*rng.integers(2, 13))),
            "source_ip": rng.choice(ips),
            "session_id": f"SID-{rng.integers(1001, 1030)}",
            "event": kind,
            "request_rate": rate,
            "failed_logins": failed,
            "bytes_out": int(rng.lognormal(8.2 if "transfer" not in kind.lower() else 12, 1.0)),
            "unique_paths": int(rng.integers(1, 3) if label == "Normal" else rng.integers(4, 40)),
            "label": label,
            "risk_score": risk,
            "severity": severity_from_risk(risk),
            "status": "Normal" if risk < 35 else ("Investigate" if risk >= 65 else "Monitoring"),
            "reason": (f"High request rate ({rate}/window)" if rate > 100 else
                       f"{failed} failed login attempts" if failed >= 5 else
                       "Unusual path diversity" if label != "Normal" else "Within simulated baseline")
        })
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)

def ensure_schema(df):
    out = df.copy()
    aliases = {
        "srcip": "source_ip", "src_ip": "source_ip", "source": "source_ip", "source address": "source_ip",
        "flow_duration": "flow_duration", "dur": "flow_duration", "duration": "flow_duration",
        "tot_fwd_pkts": "total_fwd_packets", "total_fwd_packets": "total_fwd_packets",
        "tot_bwd_pkts": "total_bwd_packets", "total_bwd_packets": "total_bwd_packets",
        "totlen_fwd_pkts": "total_fwd_bytes", "total_length_of_fwd_packets": "total_fwd_bytes",
        "totlen_bwd_pkts": "total_bwd_bytes", "total_length_of_bwd_packets": "total_bwd_bytes",
        "label": "label", "attack": "label", "attack_cat": "label", "class": "label", "target": "label",
        "protocol": "protocol", "destination_port": "destination_port", "dst_port": "destination_port",
        "dport": "destination_port", "packet_length_mean": "packet_length_mean", "pkt_len_mean": "packet_length_mean",
        "failed_logins": "failed_logins", "request_rate": "request_rate", "session_id": "session_id",
        "timestamp": "timestamp", "time": "timestamp", "bytes_out": "bytes_out"
    }
    rename = {}
    for col in out.columns:
        key = str(col).strip().lower().replace("-", "_").replace(" ", "_")
        if key in aliases: rename[col] = aliases[key]
    out = out.rename(columns=rename)
    return out

def feature_frame(df):
    numeric = df.select_dtypes(include=[np.number, "bool"]).copy()
    # remove common target leakage fields and IDs
    drop_words = ("label", "target", "attack", "class", "id", "timestamp", "time")
    numeric = numeric[[c for c in numeric.columns if not any(w in c.lower() for w in drop_words)]]
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.notna().any()]
    return numeric

def train_models(df, label_col):
    X = feature_frame(df)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns were found. Upload a dataset with numeric network/session features.")
    labels = df[label_col].astype(str).fillna("Unknown")
    counts = labels.value_counts()
    keep = counts[counts >= 2].index
    mask = labels.isin(keep)
    X, labels = X.loc[mask], labels.loc[mask]
    if labels.nunique() < 2 or len(labels) < 10:
        raise ValueError("Need at least 10 usable rows and 2 label classes with at least 2 rows each.")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=180, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "y_test": y_test, "pred": pred, "labels": list(le.classes_),
        "test_size": len(y_test), "feature_names": list(X.columns)
    }
    # Isolation Forest is trained on rows predicted/labelled normal when enough are available.
    normal_names = [i for i, c in enumerate(le.classes_) if any(x in c.lower() for x in ["normal", "benign"])]
    normal_mask = np.isin(y, normal_names) if normal_names else np.zeros(len(y), dtype=bool)
    anomaly_X = X.loc[mask]
    anomaly_fit = X.loc[mask & df[label_col].astype(str).str.lower().isin(["normal", "benign", "0"])]
    if len(anomaly_fit) < 20:
        anomaly_fit = X_train
    iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", IsolationForest(n_estimators=150, contamination="auto", random_state=42))
    ])
    iso.fit(anomaly_fit)
    model = clf.named_steps["model"]
    importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    return clf, iso, le, metrics, importances, list(X.columns)

def make_prediction_table(df, clf, iso, label_encoder, feature_names):
    work = df.copy()
    X = feature_frame(work)
    for col in feature_names:
        if col not in X.columns: X[col] = np.nan
    X = X[feature_names]
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    conf = probs.max(axis=1)
    # IsolationForest: -1 indicates anomaly
    anomaly = iso.predict(X) == -1
    pred_labels = label_encoder.inverse_transform(preds)
    risk = []
    reasons = []
    for i, (_, row) in enumerate(work.iterrows()):
        score = int(np.clip((1-conf[i])*45 + (35 if anomaly[i] else 0), 0, 100))
        predtext = str(pred_labels[i]).lower()
        if any(x in predtext for x in ["attack", "dos", "ddos", "probe", "exploit", "bot", "malicious", "intrusion", "recon"]):
            score = max(score, 72)
        if "normal" in predtext or "benign" in predtext:
            score = min(score, 32) if not anomaly[i] else max(score, 48)
        risk.append(score)
        feats = []
        for c in ["failed_logins", "request_rate", "unique_paths", "bytes_out", "flow_duration", "total_fwd_packets"]:
            if c in work.columns and pd.notna(row.get(c)):
                v = row.get(c)
                if c == "failed_logins" and v >= 5: feats.append(f"{int(v)} failed logins")
                elif c == "request_rate" and v >= 100: feats.append(f"high request rate ({v:g})")
                elif c == "unique_paths" and v >= 10: feats.append(f"high path diversity ({v:g})")
                elif c == "bytes_out" and v >= work[c].quantile(.95): feats.append("large outbound data volume")
        if anomaly[i]: feats.append("behaviour differs from learned baseline")
        if not feats: feats.append("model prediction and feature pattern")
        reasons.append("; ".join(feats[:3]))
    work["predicted_class"] = pred_labels
    work["model_confidence"] = np.round(conf, 3)
    work["anomaly_flag"] = anomaly
    work["risk_score"] = risk
    work["severity"] = [severity_from_risk(x) for x in risk]
    work["explanation"] = reasons
    work["status"] = ["Investigate" if x >= 65 else ("Monitoring" if x >= 35 else "Normal") for x in risk]
    return work

def download_csv(df, filename="sentinelai_results.csv"):
    return df.to_csv(index=False).encode("utf-8")

if "events" not in st.session_state:
    st.session_state.events = sample_events()
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "Demo simulation"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="eyebrow">Security Operations Center</div>', unsafe_allow_html=True)
    st.markdown("# 🛡️ SentinelAI")
    st.caption("Behavioural intrusion detection")
    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "Overview", "Event Explorer", "Session Analysis",
        "AI Detection Lab", "Alerts & Response", "Data & Ingestion", "About"
    ], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Operating mode**")
    st.caption(st.session_state.data_mode)
    st.markdown('<span class="pill">PROTOTYPE BUILD</span>', unsafe_allow_html=True)
    st.caption("No real network traffic is monitored by default.")

events = st.session_state.events.copy()
if "severity" not in events.columns:
    events["severity"] = events.get("risk_score", pd.Series([0]*len(events))).apply(severity_from_risk)

# ---------- Shared header ----------
st.markdown('<div class="eyebrow">SENTINELAI / THREAT INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero"><h1 style="margin:0">{" ".join(page.split(" ")[:])}</h1><div class="small-muted">Detect behavioural anomalies. Correlate events. Explain risk.</div></div>', unsafe_allow_html=True)

# ---------- Overview ----------
if page == "Overview":
    c1,c2,c3,c4 = st.columns(4)
    total = len(events)
    critical = int((events["severity"] == "Critical").sum()) if "severity" in events else 0
    high = int(events["severity"].isin(["High","Critical"]).sum()) if "severity" in events else 0
    sessions = events["session_id"].nunique() if "session_id" in events else 0
    c1.metric("Events Analysed", f"{total:,}")
    c2.metric("High / Critical", f"{high:,}", help="Events at high or critical severity")
    c3.metric("Critical Alerts", f"{critical:,}")
    c4.metric("Sessions Observed", f"{sessions:,}")
    left,right = st.columns([1.35,1])
    with left:
        st.subheader("Threat severity distribution")
        sev_order = ["Critical","High","Medium","Low"]
        counts = events["severity"].value_counts().reindex(sev_order, fill_value=0).reset_index()
        counts.columns = ["Severity","Events"]
        fig = px.bar(counts, x="Severity", y="Events", color="Severity",
                     color_discrete_map=SEVERITY_COLORS, category_orders={"Severity":sev_order})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, margin=dict(l=5,r=5,t=15,b=5), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Risk score overview")
        risk_col = "risk_score" if "risk_score" in events else None
        if risk_col:
            fig2 = go.Figure(go.Indicator(mode="gauge+number", value=float(events[risk_col].mean()),
                title={"text":"Mean risk score (0–100)"},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":"#67e8f9"},
                       "steps":[{"range":[0,35],"color":"#123a4b"},{"range":[35,65],"color":"#5b4a19"},
                                {"range":[65,85],"color":"#743b20"},{"range":[85,100],"color":"#702b3e"}]}))
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=15,r=15,t=45,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Risk scores will appear after inference.")
    st.subheader("Priority queue")
    if "risk_score" in events:
        top = events.sort_values("risk_score", ascending=False).head(8)
    else: top = events.head(8)
    show_cols = [c for c in ["timestamp","source_ip","session_id","event","predicted_class","label","risk_score","severity","explanation","reason","status"] if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True, hide_index=True)
    st.caption(f"Data source: {st.session_state.data_mode}. Scores and alerts depend on the selected data/model mode.")

# ---------- Event Explorer ----------
elif page == "Event Explorer":
    st.subheader("Search and filter security events")
    a,b,c = st.columns(3)
    sev_opts = ["All"] + [s for s in ["Critical","High","Medium","Low"] if s in events.get("severity", pd.Series(dtype=str)).unique()]
    sev = a.selectbox("Severity", sev_opts)
    search = b.text_input("Search event, IP, session or class")
    min_risk = c.slider("Minimum risk score", 0, 100, 0)
    filtered = events.copy()
    if sev != "All" and "severity" in filtered: filtered = filtered[filtered["severity"] == sev]
    if "risk_score" in filtered: filtered = filtered[filtered["risk_score"] >= min_risk]
    if search:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]
    st.write(f"**{len(filtered):,}** event(s) match your filters.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Export filtered events (CSV)", download_csv(filtered), "sentinelai_filtered_events.csv", "text/csv")

# ---------- Session Analysis ----------
elif page == "Session Analysis":
    st.subheader("Session-level behavioural analysis")
    if "session_id" not in events.columns:
        st.warning("No session_id field is present. Use demo data or upload data with a session identifier.")
    else:
        if "risk_score" not in events: events["risk_score"] = 0
        agg = {"risk_score":"max"}
        for col in ["request_rate","failed_logins","bytes_out","unique_paths"]:
            if col in events: agg[col] = "sum" if col in ["failed_logins","bytes_out"] else "max"
        if "source_ip" in events: agg["source_ip"] = "first"
        if "severity" in events: agg["severity"] = lambda x: x.iloc[0]
        if "predicted_class" in events: agg["predicted_class"] = lambda x: ", ".join(pd.Series(x).dropna().astype(str).unique()[:3])
        summary = events.groupby("session_id", dropna=False).agg(agg).reset_index().sort_values("risk_score", ascending=False)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect session", summary["session_id"].astype(str).tolist())
        detail = events[events["session_id"].astype(str) == str(selected)]
        st.markdown(f"#### Session `{selected}`")
        st.write(f"{len(detail)} event(s) associated with this session in the current data.")
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.info("Correlation is based on the available session_id field. If the source dataset has no session IDs, genuine session reconstruction requires additional telemetry and a defined grouping strategy.")

# ---------- AI Detection Lab ----------
elif page == "AI Detection Lab":
    st.subheader("Train and evaluate a baseline model")
    st.write("Upload a labelled CSV in **Data & Ingestion**, select the label column, then train a Random Forest classifier and an Isolation Forest anomaly detector.")
    if st.session_state.trained and st.session_state.model_bundle:
        clf, iso, le, metrics, importances, feature_names = st.session_state.model_bundle
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall", f"{metrics['recall']:.1%}")
        m4.metric("F1 score", f"{metrics['f1']:.1%}")
        st.caption(f"Hold-out test set: {metrics['test_size']} rows. These metrics apply only to this split and dataset; they do not guarantee performance on unseen real-world attacks.")
        l,r = st.columns([1,1])
        with l:
            st.markdown("#### Top feature importances")
            topimp = importances.head(15).sort_values("Importance")
            fig = px.bar(topimp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#164e63","#22d3ee"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=5,r=5,t=10,b=5), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.markdown("#### Confusion matrix")
            cm = confusion_matrix(metrics["y_test"], metrics["pred"], labels=np.arange(len(metrics["labels"])))
            figcm = px.imshow(cm, text_auto=True, x=metrics["labels"], y=metrics["labels"], aspect="auto", color_continuous_scale="Teal")
            figcm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=420, xaxis_title="Predicted", yaxis_title="Actual", margin=dict(l=5,r=5,t=10,b=5))
            st.plotly_chart(figcm, use_container_width=True)
        if st.button("Run model on current dataset"):
            source_df = st.session_state.get("uploaded_df")
            if source_df is None:
                st.warning("Upload a dataset first.")
            else:
                with st.spinner("Scoring rows..."):
                    scored = make_prediction_table(source_df, clf, iso, le, feature_names)
                    st.session_state.events = scored
                    st.session_state.data_mode = "Model predictions on uploaded CSV"
                st.success(f"Scored {len(scored):,} rows. Visit Overview and Event Explorer to inspect the results.")
                st.dataframe(scored.head(30), use_container_width=True)
                st.download_button("Download model predictions", download_csv(scored), "sentinelai_model_predictions.csv", "text/csv")
    else:
        st.info("No model trained in this session yet.")
        st.markdown("**Baseline workflow**")
        st.markdown("1. Upload a labelled CSV in Data & Ingestion.\n2. Choose the correct label column.\n3. Train the baseline model.\n4. Review metrics and feature importances.\n5. Score rows and inspect alerts.")
    st.warning("Important: Random Forest is supervised and only learns patterns represented in the training data. Isolation Forest flags deviations from its learned baseline; neither guarantees detection of zero-day attacks.")

# ---------- Alerts & Response ----------
elif page == "Alerts & Response":
    st.subheader("Alert triage")
    if "risk_score" in events:
        alerts = events[events["risk_score"] >= 35].sort_values("risk_score", ascending=False).copy()
    else:
        alerts = events.iloc[0:0].copy()
    if alerts.empty:
        st.success("No events meet the current alert threshold.")
    else:
        st.write(f"**{len(alerts)}** event(s) need review.")
        for _, row in alerts.head(30).iterrows():
            severity = str(row.get("severity", severity_from_risk(row.get("risk_score", 0))))
            label = str(row.get("predicted_class", row.get("label", row.get("event", "Security event"))))
            title = f"{severity} · Risk {row.get('risk_score', '—')} · {label}"
            with st.expander(title):
                st.write(f"**Source IP:** {row.get('source_ip', 'Not available')}")
                st.write(f"**Session:** {row.get('session_id', 'Not available')}")
                st.write(f"**Model confidence:** {row.get('model_confidence', 'Not available')}")
                st.write(f"**Explanation:** {row.get('explanation', row.get('reason', 'No explanation available'))}")
                st.write("**Suggested response:**")
                if severity in ["Critical", "High"]:
                    st.write("- Validate the alert against raw logs and related sessions.\n- Consider rate limiting or temporarily restricting the source only after analyst verification.\n- Preserve relevant logs for investigation.")
                else:
                    st.write("- Monitor related events and compare against normal baseline.\n- Escalate if activity repeats or correlates with other indicators.")
        st.download_button("Export alert queue (CSV)", download_csv(alerts), "sentinelai_alert_queue.csv", "text/csv")
    st.caption("This prototype does not block IPs, change firewall rules, or notify a real owner. Those actions require carefully authorised integrations.")

# ---------- Data & Ingestion ----------
elif page == "Data & Ingestion":
    st.subheader("Bring your own network or session dataset")
    st.markdown("Upload a CSV file. The app will inspect its schema and allow training on a label column. Keep sensitive logs anonymised and do not upload secrets or personal data.")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
            raw = ensure_schema(raw)
            st.session_state.uploaded_df = raw
            st.session_state.data_mode = "Uploaded dataset (not live traffic)"
            st.success(f"Loaded {len(raw):,} rows and {len(raw.columns):,} columns.")
            st.markdown("#### Dataset preview")
            st.dataframe(raw.head(12), use_container_width=True)
            st.markdown("#### Column profile")
            profile = pd.DataFrame({"Column": raw.columns, "Dtype": [str(x) for x in raw.dtypes],
                                    "Missing %": [round(raw[c].isna().mean()*100, 2) for c in raw.columns],
                                    "Unique values": [raw[c].nunique(dropna=True) for c in raw.columns]})
            st.dataframe(profile, use_container_width=True, hide_index=True)
            label_candidates = [c for c in raw.columns if c.lower() in ["label","attack_cat","attack","class","target"]]
            default_label = label_candidates[0] if label_candidates else raw.columns[-1]
            label_col = st.selectbox("Label column for supervised training", raw.columns.tolist(), index=raw.columns.tolist().index(default_label))
            if st.button("Train baseline models", type="primary"):
                try:
                    with st.spinner("Training Random Forest and Isolation Forest..."):
                        bundle = train_models(raw, label_col)
                    st.session_state.model_bundle = bundle
                    st.session_state.trained = True
                    st.session_state.label_col = label_col
                    st.success("Models trained for this session. Open AI Detection Lab to review evaluation.")
                except Exception as e:
                    st.error(f"Could not train model: {e}")
            st.download_button("Download normalised dataset preview", download_csv(raw.head(1000)), "sentinelai_dataset_preview.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")
    else:
        st.markdown("#### Try the included sample")
        st.write("The app starts in demo simulation mode so you can explore the UI before uploading a dataset.")
        st.write("Suggested public datasets from the concept note include UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019, ToN-IoT, BoT-IoT, Edge-IIoTset, and CICIoT2023.")
        if st.button("Restore demo simulation"):
            st.session_state.events = sample_events()
            st.session_state.data_mode = "Demo simulation"
            st.session_state.trained = False
            st.session_state.model_bundle = None
            st.success("Demo data restored.")

# ---------- About ----------
elif page == "About":
    st.subheader("Project scope")
    st.markdown("""
**SentinelAI** is a prototype for behavioural intrusion detection. It is designed to demonstrate a workflow that can combine network features, model-based classification, anomaly scoring, session-level grouping, and explainable alert triage.

**Implemented in this build**
- Interactive Streamlit dashboard and event explorer
- Simulated event stream for interface demonstrations
- CSV dataset ingestion and schema profiling
- Supervised Random Forest baseline for labelled numeric features
- Isolation Forest baseline for anomaly scoring
- Feature importance, hold-out evaluation metrics, confusion matrix
- Risk/severity mapping, human-readable reasons, session grouping
- CSV export for events and alert queue

**Not yet implemented**
- Live packet capture or direct website/API instrumentation
- Automatic reconstruction of multi-step sessions from raw logs lacking session IDs
- Production alert delivery (email/SMS/webhook)
- Automated blocking or response actions
- Robust validation against unseen attack families across independent datasets

**Evaluation guidance**
Use a held-out test split for a baseline, then test generalisation with a separate attack category or dataset that was excluded from training. Avoid reporting only accuracy when classes are imbalanced; include precision, recall, F1, confusion matrix, and limitations.
""")
    st.markdown("---")
    st.caption("SentinelAI • Hackathon prototype • Human review required for security decisions.")
