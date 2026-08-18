import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# App config & defaults
# ----------------------------
st.set_page_config(page_title="Smart Energy Meter (Bengaluru) — Device Attribution & Optimization", layout="wide")
st.title("🔌 Smart Energy Meter With Real Time Usage Monitoring and Bill Prediction")

# Defaults for Bengaluru / BESCOM
COST_PER_UNIT_DEFAULT = 5.90      # ₹ / kWh
FIXED_CHARGE_RATE_DEFAULT = 145.0 # ₹ / kW / month
SANCTIONED_LOAD_DEFAULT = 2.0     # kW

# Sidebar: tariff, upload, AND optimization settings
st.sidebar.header("⚙ Settings ")
cost_per_unit = st.sidebar.number_input("Energy charge (₹ / kWh)", value=COST_PER_UNIT_DEFAULT, min_value=0.0, step=0.01)
sanctioned_load_kw = st.sidebar.number_input("Sanctioned load (kW)", value=SANCTIONED_LOAD_DEFAULT, min_value=0.0, step=0.5)
fixed_charge_rate = st.sidebar.number_input("Fixed charge rate (₹ / kW / month)", value=FIXED_CHARGE_RATE_DEFAULT, min_value=0.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Optimization Settings")

# Idle detection: device considered 'idle' if its power is above 0 and below idle_pct * device_max for at least idle_min_minutes
idle_min_minutes = st.sidebar.number_input("Idle detection: minimum consecutive minutes", value=10, min_value=1, max_value=180, step=1)
idle_pct = st.sidebar.slider("Idle threshold (percent of device peak)", min_value=1, max_value=50, value=15, step=1)

# Overuse detection
overuse_multiplier = st.sidebar.slider("Overuse multiplier (compared to device typical max)", min_value=1.0, max_value=3.0, value=1.25, step=0.05)

# Unusual time-of-use detection
st.sidebar.markdown("**Unusual hours (for alerts)**")
unusual_start = st.sidebar.number_input("Unusual start hour (0-23)", value=0, min_value=0, max_value=23, step=1)
unusual_end = st.sidebar.number_input("Unusual end hour (0-23)", value=6, min_value=0, max_value=23, step=1)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Energy CSV (must have date_time & global_active_power)", type=["csv"])

# ----------------------------
# Helpers: load + preprocess
# ----------------------------
def load_and_prepare_data(file):
    df = pd.read_csv(file)
    # normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    timestamp_col = "date_time"
    power_col = "global_active_power"
    # ensure timestamp exists
    if timestamp_col not in df.columns or power_col not in df.columns:
        raise ValueError(f"CSV must contain columns: '{timestamp_col}' and '{power_col}'")
    # parse time, drop bad rows
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, power_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    # per-minute kWh conversion for global power
    df["kwh"] = df[power_col] / 1000.0 / 60.0
    # detect device columns (ending with _w)
    device_cols = [c for c in df.columns if c.endswith("_w") and c != power_col]
    return df, timestamp_col, power_col, device_cols

# ----------------------------
# Billing utilities (unchanged)
# ----------------------------
def monthly_energy_and_bill(df, timestamp_col, cost_per_unit, sanctioned_load_kw, fixed_charge_rate, device_cols):
    monthly = df.groupby(df[timestamp_col].dt.to_period("M"))["kwh"].sum().reset_index()
    monthly["month_str"] = monthly[timestamp_col].astype(str)
    monthly = monthly.rename(columns={"kwh": "units_kwh"})
    monthly["energy_charge"] = monthly["units_kwh"] * cost_per_unit
    monthly["fixed_charge"] = sanctioned_load_kw * fixed_charge_rate
    monthly["total_bill"] = monthly["energy_charge"] + monthly["fixed_charge"]
    monthly = monthly[["month_str", "units_kwh", "energy_charge", "fixed_charge", "total_bill"]].round(2)

    device_monthly = None
    if device_cols:
        device_kwh = {}
        for d in device_cols:
            device_kwh[d] = (df[d] / 1000.0 / 60.0).groupby(df[timestamp_col].dt.to_period("M")).sum()
        device_monthly = pd.DataFrame(device_kwh).reset_index().rename(columns={timestamp_col: "month"})
        device_monthly["month_str"] = device_monthly[device_monthly.columns[0]].astype(str)
        cols = ["month_str"] + [c for c in device_monthly.columns if c not in ["month_str", device_monthly.columns[0]]]
        device_monthly = device_monthly[cols].round(3)

    return monthly, device_monthly

# ----------------------------
# ML / Prediction (unchanged)
# ----------------------------
def get_monthly_prediction(monthly_df, cost_per_unit, sanctioned_load_kw, fixed_charge_rate):
    if monthly_df is None or len(monthly_df) < 1:
        return None, None
    X = np.arange(len(monthly_df)).reshape(-1,1)
    y = monthly_df["units_kwh"].values
    model = LinearRegression().fit(X, y)
    next_index = np.array([[len(X)]])
    predicted_kwh = float(model.predict(next_index)[0])
    predicted_energy_charge = predicted_kwh * cost_per_unit
    predicted_fixed = sanctioned_load_kw * fixed_charge_rate
    predicted_total_bill = predicted_energy_charge + predicted_fixed
    return predicted_kwh, predicted_total_bill

# ----------------------------
# Anomaly detection + attribution (unchanged)
# ----------------------------
def detect_anomalies_and_attribute(df, power_col, device_cols):
    mean = df[power_col].mean()
    std = df[power_col].std() if df[power_col].std() != 0 else 1.0
    df["z_score"] = (df[power_col] - mean) / std
    df["status"] = df["z_score"].apply(lambda x: "⚠ Abnormal" if abs(x) > 2 else "Normal")
    if device_cols:
        eps = 1e-6
        df_dev = df[device_cols].fillna(0)
        total = df[power_col].replace(0, eps)
        share = (df_dev.div(total, axis=0) * 100).fillna(0)
        primary_device = share.idxmax(axis=1)
        top3 = []
        for i in range(len(df)):
            row_shares = share.iloc[i].sort_values(ascending=False)
            top = row_shares[row_shares > 0].head(3)
            top_list = [f"{dev}({row_shares[dev]:.0f}%)" for dev in top.index]
            top3.append(", ".join(top_list) if len(top_list)>0 else "")
        df["primary_device"] = primary_device
        df["top3_devices"] = top3
    else:
        df["primary_device"] = ""
        df["top3_devices"] = ""
    return df

# ----------------------------
# ============ NEW: Optimization helpers ============
# ----------------------------
def _consecutive_runs(bool_series):
    # returns list of (start_idx, end_idx) inclusive for True runs
    runs = []
    in_run = False
    start = None
    for i, v in enumerate(bool_series):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            runs.append((start, i-1))
            in_run = False
    if in_run:
        runs.append((start, len(bool_series)-1))
    return runs

def detect_idle_waste(df, timestamp_col, device_cols, idle_pct, idle_min_minutes):
    """
    Identify idle runs per device:
     - idle threshold = idle_pct% of device 99th-percentile (approx peak)
     - consider run if consecutive minutes >= idle_min_minutes and power>0 and < threshold
    Returns a DataFrame of idle events with device, start, end, minutes, wasted_kwh.
    """
    events = []
    for dev in device_cols:
        series = df[dev].fillna(0).astype(float).values
        if series.sum() == 0:
            continue
        peak = np.percentile(series, 99)
        idle_thresh = (idle_pct/100.0) * max(peak, 1.0)
        # boolean series: idle condition
        cond = (series > 0) & (series <= idle_thresh)
        runs = _consecutive_runs(cond)
        for (s,e) in runs:
            minutes = e - s + 1
            if minutes >= idle_min_minutes:
                # wasted energy in kWh = sum(watts)/1000/60
                wasted_kwh = series[s:e+1].sum() / 1000.0 / 60.0
                events.append({
                    "device": dev,
                    "start_time": df[timestamp_col].iloc[s],
                    "end_time": df[timestamp_col].iloc[e],
                    "minutes": minutes,
                    "wasted_kwh": wasted_kwh
                })
    if len(events) == 0:
        return pd.DataFrame(columns=["device","start_time","end_time","minutes","wasted_kwh"])
    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values(["wasted_kwh"], ascending=False).reset_index(drop=True)
    return events_df

def detect_overuse_events(df, device_cols, overuse_multiplier):
    """
    Detect rows where device power exceeds overuse_multiplier * device 95th-percentile
    Returns DataFrame with device, timestamp, power, threshold.
    """
    rows = []
    for dev in device_cols:
        series = df[dev].fillna(0).astype(float)
        if series.sum() == 0:
            continue
        thresh = np.percentile(series, 95) * overuse_multiplier
        mask = series > thresh
        if mask.any():
            temp = df.loc[mask, ["date_time", dev]].copy()
            temp = temp.rename(columns={dev: "power"})
            temp["device"] = dev
            temp["threshold"] = thresh
            rows.append(temp[["date_time","device","power","threshold"]])
    if len(rows) == 0:
        return pd.DataFrame(columns=["date_time","device","power","threshold"])
    return pd.concat(rows).sort_values("date_time").reset_index(drop=True)

def detect_unusual_time_use(df, device_cols, start_hour, end_hour):
    """
    Flag device rows that are active during the unusual window [start_hour, end_hour].
    Returns DataFrame with date_time, device, power.
    """
    rows = []
    for dev in device_cols:
        ser = df[[ "date_time", dev ]].copy()
        ser["hour"] = ser["date_time"].dt.hour
        mask = (ser["hour"] >= start_hour) & (ser["hour"] <= end_hour) & (ser[dev] > 0)
        if mask.any():
            temp = ser.loc[mask, ["date_time", dev]].copy()
            temp = temp.rename(columns={dev: "power"})
            temp["device"] = dev
            rows.append(temp[["date_time","device","power"]])
    if len(rows) == 0:
        return pd.DataFrame(columns=["date_time","device","power"])
    return pd.concat(rows).sort_values("date_time").reset_index(drop=True)

# ----------------------------
# Main UI logic (original flow kept)
# ----------------------------
if uploaded_file is None:
    st.info("Upload CSV file (date_time, global_active_power). If your CSV includes device columns use them for device attribution.")
    st.stop()

# load
try:
    df, timestamp_col, power_col, device_cols = load_and_prepare_data(uploaded_file)
except Exception as e:
    st.error(f"Failed loading file: {e}")
    st.stop()

# show basic metrics
total_energy = df["kwh"].sum()
est_bill_flat = total_energy * cost_per_unit
st.subheader("⚡ Energy Summary")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total energy (year)", f"{total_energy:.2f} kWh")
with c2:
    st.metric("Estimated bill for the year", f"₹ {est_bill_flat:.2f}")
with c3:
    st.metric("Sanctioned load (kW)", f"{sanctioned_load_kw:.1f} kW")

# monthly billing
monthly, device_monthly = monthly_energy_and_bill(df, timestamp_col, cost_per_unit, sanctioned_load_kw, fixed_charge_rate, device_cols)
st.subheader("📊 Monthly Energy & Bill ")
st.dataframe(monthly, use_container_width=True)

# device monthly if available
if device_monthly is not None and not device_monthly.empty:
    st.subheader("📦 Energy Consumed Per Device Monthly (kWh)")
    st.dataframe(device_monthly, use_container_width=True)
    # plot per-device for last month
    try:
        last_month = device_monthly.iloc[-1]
        per_dev = last_month.drop("month_str").to_dict()
        dev_names = [k for k in per_dev.keys()]
        dev_vals = [float(per_dev[k]) for k in dev_names]
        fig = px.bar(x=dev_names, y=dev_vals, labels={"x":"Device", "y":"kWh"}, title=f"Energy Consumed Per Device (last month = {last_month['month_str']})")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# monthly prediction (ML)
pred_kwh, pred_total_bill = get_monthly_prediction(monthly, cost_per_unit, sanctioned_load_kw, fixed_charge_rate)
st.subheader("📈 Bill Predicted For Next Month")
if pred_kwh is not None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted units (kWh)", f"{pred_kwh:.2f}")
    with col_b:
        st.metric("Predicted bill (incl fixed)", f"₹ {pred_total_bill:.2f}")
else:
    st.info("Not enough monthly data to predict next month.")

# anomaly detection + attribution
st.subheader("🔍 Anomalies Detected ")
df = detect_anomalies_and_attribute(df, power_col, device_cols)
anomalies = df[df["status"] != "Normal"].copy()
st.write(f"Total anomalies found: **{len(anomalies)}**")
if len(anomalies) == 0:
    st.info("No anomalies detected in dataset.")
else:
    cols_to_show = [timestamp_col, power_col, "z_score", "status", "primary_device", "top3_devices"]
    st.dataframe(anomalies[cols_to_show].sort_values(timestamp_col, ascending=False).head(200), use_container_width=True)
    if device_cols:
        summary = anomalies["primary_device"].value_counts().reset_index()
        summary.columns = ["device", "count"]
        st.subheader("Anomaly attribution summary (primary device)")
        st.dataframe(summary, use_container_width=True)
        fig_sum = px.pie(summary, names="device", values="count", title="Anomalies by Primary Device")
        st.plotly_chart(fig_sum, use_container_width=True)
    else:
        st.info("No device columns found to attribute anomalies. If your CSV includes per-device watts (e.g., fridge_w), they must end with '_w'.")

# ----------------------------
# ============ NEW: Power Optimization section ============
# ----------------------------
st.subheader("🔧 Power Optimization — Idle / Overuse / Unusual-time Detection")
if not device_cols:
    st.info("No per-device columns detected. To use optimization features, include device columns named like fridge_w, ac_w, tv_w, etc.")
else:
    with st.spinner("Analyzing device idle/waste and overuse..."):
        # Idle waste detection
        idle_events = detect_idle_waste(df, timestamp_col, device_cols, idle_pct=idle_pct, idle_min_minutes=int(idle_min_minutes))
        # Overuse detection
        overuse_df = detect_overuse_events(df, device_cols, overuse_multiplier=overuse_multiplier)
        # Unusual time-of-use
        unusual_df = detect_unusual_time_use(df, device_cols, unusual_start, unusual_end)

        # Summaries
        if idle_events.empty:
            st.write("No sustained idle-waste events detected with current thresholds.")
        else:
            st.markdown("**Idle / Wasteful Devices**")
            # compute estimated savings (₹)
            idle_events["estimated_savings_rs"] = (idle_events["wasted_kwh"] * cost_per_unit).round(2)
            st.dataframe(idle_events[["device","start_time","end_time","minutes","wasted_kwh","estimated_savings_rs"]].head(200), use_container_width=True)

            # Top wasteful devices bar
            top_waste = idle_events.groupby("device")["wasted_kwh"].sum().sort_values(ascending=False).reset_index()
            top_waste["estimated_savings_rs"] = (top_waste["wasted_kwh"] * cost_per_unit).round(2)
            st.markdown("**Top wasteful devices (kWh)**")
            st.dataframe(top_waste, use_container_width=True)
            fig_waste = px.bar(top_waste, x="device", y="wasted_kwh", labels={"wasted_kwh":"Wasted kWh"}, title="Energy Wasted Per Device (kWh)")
            st.plotly_chart(fig_waste, use_container_width=True)

            total_wasted_kwh = top_waste["wasted_kwh"].sum()
            total_potential_savings = (total_wasted_kwh * cost_per_unit)
            st.success(f"Estimated monthly savings if idle-waste is eliminated: ₹ {total_potential_savings:.2f}")

        # Overuse table
        if overuse_df.empty:
            st.write("No device overuse events detected with current thresholds.")
        else:
            st.markdown("**Overused Devices(kWh)**")
            st.dataframe(overuse_df.head(200), use_container_width=True)

        # Unusual time-of-use
        if unusual_df.empty:
            st.write("No unusual time-of-use events detected in configured window.")
        else:
            st.markdown(f"**Devices active between {unusual_start}:00 and {unusual_end}:00**")
            st.dataframe(unusual_df.head(200), use_container_width=True)

        # Tips based on findings
        st.markdown("**Optimization Suggestions**")
        tips = []
        if not idle_events.empty:
            # recommend top device
            top_dev = top_waste.iloc[0]["device"]
            est = top_waste.iloc[0]["estimated_savings_rs"]
            tips.append(f"Turn off or auto-schedule **{top_dev}** when not used — could save ~₹{est:.0f} for the observed period.")
        if not overuse_df.empty:
            devs = overuse_df["device"].unique().tolist()
            tips.append(f"Check devices with repeated overuse: {', '.join(devs)} — may indicate faulty appliance or heavy setting.")
        if not unusual_df.empty:
            tips.append("Consider timers or occupancy-based switching for devices active in unusual hours.")
        if not tips:
            st.write("No strong optimization actions detected — good job! You can relax thresholds in the sidebar to be more aggressive.")
        else:
            for t in tips:
                st.write("• " + t)

# ----------------------------
# Smart insights (unchanged)
# ----------------------------
st.subheader("🧠 Smart Insights & Suggestions")
try:
    df["hour"] = df[timestamp_col].dt.hour
    hourly = df.groupby("hour")["kwh"].sum()
    if not hourly.empty:
        peak_hour = int(hourly.idxmax())
        peak_energy = float(hourly.max())
        avg_daily = float(df.groupby(df[timestamp_col].dt.date)["kwh"].sum().mean())
        st.write(f"• Peak aggregated hour: **{peak_hour}:00** (approx {peak_energy:.2f} kWh).")
        st.write(f"• Average daily consumption: **{avg_daily:.2f} kWh/day**.")
        st.write(f"• Fixed monthly charge (assumed): **₹ {sanctioned_load_kw * fixed_charge_rate:.2f}**.")
        if device_cols:
            dev_totals = df[[*device_cols]].sum().sort_values(ascending=False)
            top_dev = dev_totals.index[0]
            top_dev_kwh = (dev_totals.iloc[0] / 1000.0 / 60.0)
            st.write(f"• Top consuming device overall: **{top_dev}** (~{top_dev_kwh:.2f} kWh across dataset).")
            st.write("• Tip: Check the anomaly summary above to see which devices cause spikes.")
except Exception:
    st.info("Not enough data for insights.")
