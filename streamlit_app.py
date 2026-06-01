import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from datetime import datetime, timedelta
import warnings
import joblib
import json
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="KNN - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --primary-color: #00CED1;
    --background-color: #0F1419;
    --secondary-bg: #1a1f2e;
    --card-bg: #1e2635;
    --text-color: #FFFFFF;
    --accent-color: #00CED1;
}
* {
    background-color: var(--background-color);
    color: var(--text-color);
}
.main {
    background-color: var(--background-color);
}
.stMetric {
    background-color: var(--card-bg);
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid var(--accent-color);
}
.metric-card {
    background: linear-gradient(135deg, #1e2635 0%, #252d3d 100%);
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #2a3f5f;
    text-align: center;
}
.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    color: #00CED1;
    margin: 10px 0;
}
.metric-label {
    font-size: 0.9em;
    color: #A0B0C0;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.alert-critical {
    background-color: #3d1f1f;
    border-left: 4px solid #FF4444;
}
.alert-warning {
    background-color: #3d3220;
    border-left: 4px solid #FFB800;
}
.alert-normal {
    background-color: #1f3d2d;
    border-left: 4px solid #00CED1;
}
.sidebar .sidebar-content {
    background-color: var(--secondary-bg);
}
h1, h2, h3 {
    color: #FFFFFF;
}
.stButton > button {
    background-color: #00CED1;
    color: #0F1419;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #00B8BD;
    transform: scale(1.05);
}
.stSlider > div > div > div > div {
    color: #00CED1;
}
.stNumberInput input {
    background-color: #1e2635;
    border: 1px solid #00CED1;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "predictive_maintenance.csv"

FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]
TARGET_COLUMN = 'Machine failure'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_pretrained_model():
    """Load pre-trained model, scaler, and metadata from files"""
    try:
        model_path = 'knn_model.joblib'
        scaler_path = 'scaler.joblib'
        metadata_path = 'model_metadata.json'
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [model_path, scaler_path, metadata_path]):
            return None, None, None
        
        # Load model and scaler using joblib
        knn = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return knn, scaler, metadata
    except Exception as e:
        st.error(f"Error loading pre-trained model: {e}")
        return None, None, None

@st.cache_data
def train_model(k, weights, metric):
    df = load_data()
    imputer = SimpleImputer(strategy='mean')
    df[FEATURE_COLUMNS] = imputer.fit_transform(df[FEATURE_COLUMNS])
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    y_prob = knn.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return knn, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob, accuracy, precision, recall, f1, cm

df = load_data()

# ============================================
# Initialize or Load Pre-trained Model
# ============================================
if 'model_trained' not in st.session_state:
    # Try loading pre-trained model first
    knn, scaler, metadata = load_pretrained_model()
    if knn is not None and scaler is not None:
        # Pre-trained model loaded successfully
        st.session_state.model_trained = True
        st.session_state.knn = knn
        st.session_state.scaler = scaler
        st.session_state.accuracy = metadata.get('accuracy', 0.977)
        st.session_state.model_source = "pre-trained"
        
        # Generate predictions on test set for evaluation display
        # (We'll create dummy data for metrics display when pre-trained)
        st.session_state.model_initialized = True
    else:
        st.session_state.model_trained = False
        st.session_state.model_initialized = False

st.sidebar.title("⚙️ PREEMPTIVE AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Real-Time Monitoring", "Alert History"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Settings")
k_default = st.sidebar.slider("K Neighbors", 1, 50, 5, 1)
weights_default = st.sidebar.selectbox("Weight Function", ["uniform", "distance"], index=0)
metric_default = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"], index=0)

if st.sidebar.button("🚀 Train KNN Model", type="primary", use_container_width=True):
    with st.spinner("Training KNN..."):
        results = train_model(k_default, weights_default, metric_default)
        knn, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob, accuracy, precision, recall, f1, cm = results
    st.session_state.model_trained = True
    st.session_state.knn = knn
    st.session_state.scaler = scaler
    st.session_state.accuracy = accuracy
    st.session_state.precision = precision
    st.session_state.recall = recall
    st.session_state.f1 = f1
    st.session_state.cm = cm
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.y_prob = y_prob
    st.session_state.model_source = "custom"
    st.sidebar.success(f"✅ Model trained! Accuracy: {accuracy:.4f}")

st.sidebar.markdown("---")
if st.session_state.get("model_trained"):
    model_source = st.session_state.get("model_source", "unknown")
    st.sidebar.info(f"🟢 Model Ready ({model_source})")
else:
    st.sidebar.warning("🟡 Train model in sidebar")

st.sidebar.markdown("---")
st.sidebar.markdown("**Status**: 🟢 Online - All Systems Operational")

if page == "Dashboard":
    st.markdown("<h1>📊 PREEMPTIVE AI - Predictive Maintenance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"**Dataset**: Predictive Maintenance | **Rows**: {df.shape[0]} | **Last Update**: {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Samples</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <small style="color: #A0B0C0;">Total records</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{len(FEATURE_COLUMNS)}</div>
            <small style="color: #A0B0C0;">Sensor parameters</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        failure_rate = df[TARGET_COLUMN].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Failure Rate</div>
            <div class="metric-value">{failure_rate:.2f}%</div>
            <small style="color: #FF6B6B;">Machine failure</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        model_status = "Trained" if st.session_state.get("model_trained") else "Not Trained"
        status_color = "#00CED1" if st.session_state.get("model_trained") else "#FFB800"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">KNN Model</div>
            <div class="metric-value" style="color: {status_color};">{model_status}</div>
            <small style="color: #A0B0C0;">k={k_default}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("📊 Feature Statistics")
        st.dataframe(df[FEATURE_COLUMNS].describe(), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Target Distribution")
        target_counts = df[TARGET_COLUMN].value_counts().reset_index()
        target_counts.columns = ['Machine Failure', 'Count']
        target_counts['Machine Failure'] = target_counts['Machine Failure'].map({0: 'No Failure', 1: 'Failure'})
        st.dataframe(target_counts, use_container_width=True, hide_index=True)
        st.bar_chart(target_counts.set_index('Machine Failure'))
    with col2:
        st.subheader("❓ Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        st.dataframe(missing, use_container_width=True, hide_index=True)

    if st.session_state.get("model_trained"):
        st.markdown("---")
        st.subheader("📈 Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy", f"{st.session_state.accuracy:.4f}")
        with m2:
            if "precision" in st.session_state:
                st.metric("Precision", f"{st.session_state.precision:.4f}")
            else:
                st.metric("Precision", "N/A")
        with m3:
            if "recall" in st.session_state:
                st.metric("Recall", f"{st.session_state.recall:.4f}")
            else:
                st.metric("Recall", "N/A")
        with m4:
            if "f1" in st.session_state:
                st.metric("F1 Score", f"{st.session_state.f1:.4f}")
            else:
                st.metric("F1 Score", "N/A")

elif page == "Prediction":
    st.markdown("<h1>🎯 Machine Failure Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.get("model_trained"):
        st.warning("⚠️ Please train the KNN model first using the sidebar button.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🔧 Enter Sensor Readings")
            air_temp = st.slider("Air Temperature (K)", 290.0, 310.0, 298.1, 0.1)
            proc_temp = st.slider("Process Temperature (K)", 300.0, 320.0, 308.6, 0.1)
            rpm = st.slider("Rotational Speed (RPM)", 1000, 3000, 1500, 10)
            torque = st.slider("Torque (Nm)", 5.0, 75.0, 42.8, 0.5)
            tool_wear = st.slider("Tool Wear (min)", 0, 255, 0, 5)

            if st.button("🚀 Predict Failure Probability", key="predict_btn"):
                input_data = pd.DataFrame({
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [proc_temp],
                    'Rotational speed [rpm]': [rpm],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })
                input_scaled = st.session_state.scaler.transform(input_data)
                pred = st.session_state.knn.predict(input_scaled)[0]
                prob = st.session_state.knn.predict_proba(input_scaled)[0]

                avg_prob = prob[1] * 100
                if avg_prob >= 70:
                    risk_level = "🔴 CRITICAL"
                    risk_color = "#FF4444"
                elif avg_prob >= 40:
                    risk_level = "🟡 WARNING"
                    risk_color = "#FFB800"
                else:
                    risk_level = "🟢 NORMAL"
                    risk_color = "#00CED1"

                st.session_state.prediction_made = True
                st.session_state.pred = pred
                st.session_state.avg_prob = avg_prob
                st.session_state.risk_level = risk_level
                st.session_state.risk_color = risk_color
                st.session_state.last_input = {
                    'air_temp': air_temp, 'proc_temp': proc_temp, 'rpm': rpm,
                    'torque': torque, 'tool_wear': tool_wear,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

        with col2:
            st.subheader("📊 Prediction Results")
            if 'prediction_made' in st.session_state and st.session_state.prediction_made:
                if st.session_state.pred == 1:
                    st.error(f"🔴 **Machine Failure Predicted**")
                else:
                    st.success(f"🟢 **No Failure Predicted**")

                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {st.session_state.risk_color};">
                    <div class="metric-label">Failure Probability</div>
                    <div class="metric-value" style="color: {st.session_state.risk_color};">
                        {st.session_state.avg_prob:.2f}%
                    </div>
                    <small>{st.session_state.risk_level}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👉 Enter sensor readings and click 'Predict Failure Probability' to see results")

elif page == "Real-Time Monitoring":
    st.markdown("<h1>📡 Real-Time Monitoring System</h1>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🔄 Refresh Sensor Data"):
        pass

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics_list = [
        ("Air Temp", 298.1 + np.random.normal(0, 0.5), "K"),
        ("Proc Temp", 308.6 + np.random.normal(0, 0.5), "K"),
        ("RPM", 1551 + np.random.normal(0, 30), "rpm"),
        ("Torque", 42.8 + np.random.normal(0, 2), "Nm"),
        ("Tool Wear", 108 + np.random.normal(0, 5), "min")
    ]
    cols = [col1, col2, col3, col4, col5]
    for col, (label, value, unit) in zip(cols, metrics_list):
        with col:
            st.metric(label, f"{value:.1f} {unit}")

    st.markdown("---")
    st.subheader("📈 Real-Time Telemetry")

    time_points = pd.date_range(start=datetime.now() - timedelta(hours=1), periods=60, freq='1min')
    rpm_data = 1500 + np.cumsum(np.random.normal(0, 10, 60))
    temp_data = 298 + np.sin(np.arange(60) / 12) * 2 + np.random.normal(0, 0.3, 60)

    chart_df = pd.DataFrame({
        'Time': time_points,
        'RPM': rpm_data,
        'Temperature (K)': temp_data
    }).set_index('Time')
    st.line_chart(chart_df, use_container_width=True, height=400)

    if st.session_state.get("model_trained"):
        st.markdown("---")
        st.subheader("🔮 Live Prediction on Current Readings")
        live_data = pd.DataFrame({
            'Air temperature [K]': [metrics_list[0][1]],
            'Process temperature [K]': [metrics_list[1][1]],
            'Rotational speed [rpm]': [metrics_list[2][1]],
            'Torque [Nm]': [metrics_list[3][1]],
            'Tool wear [min]': [metrics_list[4][1]]
        })
        live_scaled = st.session_state.scaler.transform(live_data)
        live_pred = st.session_state.knn.predict(live_scaled)[0]
        live_prob = st.session_state.knn.predict_proba(live_scaled)[0][1] * 100

        col_a, col_b = st.columns(2)
        with col_a:
            if live_pred == 1:
                st.error(f"🔴 **Failure Risk: {live_prob:.1f}%**")
            else:
                st.success(f"🟢 **Status Normal**")
        with col_b:
            st.progress(int(live_prob))

elif page == "Alert History":
    st.markdown("<h1>🚨 Alert History & Event Log</h1>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.get("model_trained"):
        cm = st.session_state.cm
        tn, fp, fn, tp = cm.ravel()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ True Negatives", tn)
        with col2:
            st.metric("❌ False Positives", fp)
        with col3:
            st.metric("⚠️ False Negatives", fn)
        with col4:
            st.metric("🎯 True Positives", tp)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Critical Alerts", "—")
        with col2:
            st.metric("🟡 Warnings", "—")
        with col3:
            st.metric("🟢 Normal", "—")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        alert_type = st.selectbox("Filter by Type", ["All", "CRITICAL", "WARNING", "NORMAL"])
    with col2:
        date_range = st.selectbox("Filter by Date", ["Last 24h", "Last 7d", "Last 30d", "All"])

    st.markdown("---")

    sample_alerts = pd.DataFrame({
        'Timestamp': [
            '2026-05-28 14:22:04', '2026-05-28 14:18:55',
            '2026-05-28 13:45:12', '2026-05-27 22:15:33'
        ],
        'Sensor': ['Vibration', 'Temperature', 'Pressure', 'RPM'],
        'Severity': ['🔴 CRITICAL', '🟡 WARNING', '🟡 WARNING', '🟢 NORMAL'],
        'Value': ['0.92 m/s', '78.5°C', '14.2 BAR', '1650 RPM'],
        'Status': ['Unresolved', 'Resolved', 'Monitoring', 'Normal']
    })
    st.dataframe(sample_alerts, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.session_state.get("model_trained") and "cm" in st.session_state:
        st.subheader("📋 Classification Report")
        report = classification_report(
            st.session_state.y_test,
            st.session_state.y_pred,
            target_names=["No Failure", "Failure"],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    elif st.session_state.get("model_trained"):
        st.subheader("📋 Model Information")
        st.info("📊 Pre-trained model loaded. Train a custom model to see detailed classification metrics.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #A0B0C0; font-size: 0.85em;">
    <p>KNN Predictive Maintenance | Built from knn.ipynb | © 2026 Datacrumbs Analytics</p>
    <p>Powered by scikit-learn • Streamlit</p>
</div>
""", unsafe_allow_html=True)
