"""
UAC Program - Care Load & Placement Demand Forecasting Dashboard
Run:  streamlit run app.py
Reads precomputed outputs from ./artifacts (created by pipeline.py).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
EPS = 1.0
GRID = [1, 3, 7, 14, 21, 28]
COLORS = {
    "Naive (persistence)": "#9aa0a6", "Moving average (7d)": "#b39ddb", "Moving average (14d)": "#b39ddb",
    "Exponential smoothing": "#f9a825", "ARIMA(2,1,1)": "#26a69a", "SARIMA(1,1,1)(1,0,1,7)": "#00897b",
    "Random Forest": "#5c6bc0", "Gradient Boosting": "#e53935",
}
FAMILY = {"Naive (persistence)": "Baseline", "Moving average (7d)": "Baseline", "Moving average (14d)": "Baseline",
          "Exponential smoothing": "Statistical", "ARIMA(2,1,1)": "Statistical",
          "SARIMA(1,1,1)(1,0,1,7)": "Statistical", "Random Forest": "Machine learning",
          "Gradient Boosting": "Machine learning"}

st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    if not (ART / "forecast_final.csv").exists():
        return None
    data = pd.read_csv(ART / "data_clean.csv", parse_dates=["date"]).set_index("date")
    fc = pd.read_csv(ART / "forecast_final.csv", parse_dates=["date"])
    bt = pd.read_csv(ART / "backtest.csv", parse_dates=["origin", "target_date"])
    mh = pd.read_csv(ART / "metrics_horizon.csv")
    mr = pd.read_csv(ART / "metrics_regime.csv")
    kp = pd.read_csv(ART / "kpis_by_model.csv")
    cov = pd.read_csv(ART / "coverage.csv")
    meta = json.loads((ART / "meta.json").read_text())
    return data, fc, bt, mh, mr, kp, cov, meta


def breach_probability(bt, target, model, basis, h, point, threshold):
    """P(actual > threshold) from the empirical out-of-sample error distribution at the nearest backtest horizon."""
    b = bt[(bt.target == target) & (bt.model == model) & bt.observed]
    if basis == "recent":
        b = b[b.origin >= "2025-04-01"]
    g = min(GRID, key=lambda x: abs(x - h))
    e = b[b.h == g]
    if len(e) < 8:
        e = b
    err = np.log((e["actual"] + EPS) / (e["forecast"] + EPS)).values
    sim = (point + EPS) * np.exp(err) - EPS
    return float((sim > threshold).mean())


def band(fc_df, level):
    lo, hi = ("lo80", "hi80") if level == 80 else ("lo95", "hi95")
    return fc_df[lo].values, fc_df[hi].values


bundle = load()
if bundle is None:
    st.error("Model outputs not found. Run `python pipeline.py` first to generate ./artifacts.")
    st.stop()
data, fc, bt, mh, mr, kp, cov, meta = bundle
last_date = data.index.max()
cur = float(data["hhs_care"].iloc[-1])
real = data[~data.imputed]

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Forecast controls")
horizon = st.sidebar.slider("Forecast horizon (days ahead)", 1, 28, 14)
stock_models = sorted(fc[fc.target == "hhs_care"].model.unique(), key=lambda m: list(COLORS).index(m))
mh_h = mh[(mh.target == "hhs_care")]
grid_h = min(GRID, key=lambda x: abs(x - horizon))
ranking = mh_h[mh_h.h == grid_h].sort_values("MAE")
best_at_h = ranking.iloc[0]["model"]
primary_choice = st.sidebar.selectbox("Primary model (drives intervals & risk)", ["Auto (best at horizon)"] + stock_models)
primary = best_at_h if primary_choice.startswith("Auto") else primary_choice
shown = st.sidebar.multiselect("Models to overlay", stock_models, default=[primary, "Naive (persistence)"] if primary != "Naive (persistence)" else [primary])
level = st.sidebar.radio("Confidence interval", [80, 95], horizontal=True)
basis_label = st.sidebar.radio("Uncertainty basis", ["Post-shift regime (since Apr 2025)", "Full history (2024-25)"],
                               help="Interval widths come from out-of-sample walk-forward errors. The full history includes the "
                                    "early-2025 structural drop in care load, so its intervals are much wider.")
basis = "recent" if basis_label.startswith("Post") else "full"
default_cap = int(round(cur * 1.15 / 50) * 50)
capacity = st.sidebar.number_input("Capacity threshold (children in HHS care)", 500, 15000, default_cap, 50,
                                   help="Planning threshold. Default = 115% of today's care load. Replace with the operational bed capacity.")
st.sidebar.caption(f"Data through **{last_date.date()}** · {meta['n_observed']} reported days · {meta['n_imputed']} interpolated · "
                   f"{meta['n_origins']} walk-forward origins")
if primary_choice.startswith("Auto"):
    st.sidebar.info(f"Auto-selected **{primary}** (lowest backtest MAE at the {grid_h}-day horizon).")

sel = fc[(fc.target == "hhs_care") & (fc.basis == basis)]
p_fc = sel[sel.model == primary].sort_values("h")
p_at_h = p_fc[p_fc.h == horizon].iloc[0]

# ----------------------------------------------------------------------------- header + KPIs
st.title("Care Load & Placement Demand Forecasting")
st.caption("HHS Unaccompanied Alien Children (UAC) Program · short-term predictive intelligence for capacity planning")

prob = breach_probability(bt, "hhs_care", primary, basis, horizon, p_at_h["point"], capacity)
peak_prob = max(breach_probability(bt, "hhs_care", primary, basis, int(r.h), r.point, capacity) for r in p_fc.itertuples())
chg = p_at_h["point"] / cur - 1
mrow = mh_h[(mh_h.model == primary) & (mh_h.h == grid_h)].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Children in HHS care (latest)", f"{cur:,.0f}", help=f"Reported {last_date.date()}")
c2.metric(f"Forecast in {horizon} d", f"{p_at_h['point']:,.0f}", f"{chg:+.1%}")
lo, hi = band(p_fc[p_fc.h == horizon], level)
c3.metric(f"{level}% interval", f"{lo[0]:,.0f} – {hi[0]:,.0f}")
c4.metric("Capacity breach probability", f"{prob:.0%}", f"peak over 28 d: {peak_prob:.0%}", delta_color="off")
c5.metric("Backtest accuracy at horizon", f"{mrow['Accuracy_pct']:.1f}%", f"MAPE {mrow['MAPE']:.1f}%", delta_color="off")

status = "Normal" if peak_prob < 0.10 else ("Elevated" if peak_prob < 0.40 else "High risk")
max_rise = (p_fc["point"].max() / cur - 1)
st.markdown(f"**Early-warning status: {status}** — forecast peak over the next 28 days is **{p_fc['point'].max():,.0f}** "
            f"({max_rise:+.1%} vs today); capacity threshold set at **{capacity:,}**.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Care load forecast", "Discharge demand", "Model comparison",
                                              "Scenarios", "Early warning & KPIs", "Data & method"])

# ----------------------------------------------------------------------------- tab 1
with tab1:
    hist_days = st.slider("History to display (days)", 30, 365, 120, key="hist1")
    h_df = data[data.index > last_date - pd.Timedelta(days=hist_days)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_df.index, y=h_df["hhs_care"], name="Actual (HHS care)", line=dict(color="#202124", width=2.5)))
    for m in shown:
        d = sel[(sel.model == m) & (sel.h <= horizon)].sort_values("h")
        d = pd.concat([pd.DataFrame({"date": [last_date], "point": [cur]}), d[["date", "point"]]])
        fig.add_trace(go.Scatter(x=d["date"], y=d["point"], name=m, line=dict(color=COLORS[m], width=2.2, dash="dash" if m != primary else "solid")))
    pf = p_fc[p_fc.h <= horizon]
    lo_a, hi_a = band(pf, level)
    if not np.isnan(lo_a).any():
        xs = [last_date] + list(pf["date"])
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[cur] + list(hi_a) + ([cur] + list(lo_a))[::-1],
                                 fill="toself", fillcolor="rgba(92,107,192,0.18)", line=dict(width=0),
                                 name=f"{level}% interval ({primary})", hoverinfo="skip"))
    fig.add_hline(y=capacity, line_dash="dot", line_color="#d93025", annotation_text=f"Capacity threshold {capacity:,}")
    fig.add_vline(x=last_date, line_dash="dot", line_color="#80868b")
    fig.update_layout(height=520, hovermode="x unified", yaxis_title="Children in HHS care", margin=dict(t=30),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    t = p_fc[p_fc.h.isin([1, 3, 7, 14, 21, 28])][["h", "date", "point", "lo80", "hi80", "lo95", "hi95"]].copy()
    t["breach prob."] = [breach_probability(bt, "hhs_care", primary, basis, int(h), p, capacity) for h, p in zip(t.h, t.point)]
    t["date"] = t["date"].dt.date
    st.dataframe(t.rename(columns={"h": "days ahead", "point": f"{primary}", "lo80": "80% low", "hi80": "80% high",
                                   "lo95": "95% low", "hi95": "95% high"}).style.format(
        {primary: "{:,.0f}", "80% low": "{:,.0f}", "80% high": "{:,.0f}", "95% low": "{:,.0f}", "95% high": "{:,.0f}", "breach prob.": "{:.0%}"}),
        hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Discharge (placement) demand and intake-vs-exit balance")
    flow_models = sorted(fc[fc.target == "discharged"].model.unique(), key=lambda m: list(COLORS).index(m))
    mh_d = mh[mh.target == "discharged"]
    best_d = mh_d[mh_d.h == grid_h].sort_values("MAE").iloc[0]["model"]
    fm = st.selectbox("Flow forecasting model", flow_models, index=flow_models.index(best_d),
                      help=f"Default = best discharge model at the {grid_h}-day horizon ({best_d}).")
    dsel = fc[(fc.target == "discharged") & (fc.basis == basis) & (fc.model == fm)].sort_values("h")
    tsel_models = fc[(fc.target == "transferred")].model.unique()
    tm = fm if fm in tsel_models else "Exponential smoothing"
    tsel = fc[(fc.target == "transferred") & (fc.basis == basis) & (fc.model == tm)].sort_values("h")
    hist = data[data.index > last_date - pd.Timedelta(days=90)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["discharged_7d"], name="Discharges / day (7d mean, actual)", line=dict(color="#1a73e8", width=2.2)))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["transferred_7d"], name="Transfers into HHS / day (7d mean, actual)", line=dict(color="#e8710a", width=2.2)))
    dd = dsel[dsel.h <= horizon]
    fig.add_trace(go.Scatter(x=[last_date] + list(dd["date"]), y=[hist["discharged_7d"].iloc[-1]] + list(dd["point"]), name=f"Discharge forecast ({fm})", line=dict(color="#1a73e8", dash="dash")))
    lo_a, hi_a = band(dd, level)
    if not np.isnan(lo_a).any():
        xs = list(dd["date"])
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(hi_a) + list(lo_a)[::-1], fill="toself", fillcolor="rgba(26,115,232,0.16)",
                                 line=dict(width=0), name=f"{level}% interval", hoverinfo="skip"))
    td = tsel[tsel.h <= horizon]
    fig.add_trace(go.Scatter(x=[last_date] + list(td["date"]), y=[hist["transferred_7d"].iloc[-1]] + list(td["point"]), name=f"Transfer forecast ({tm})", line=dict(color="#e8710a", dash="dash")))
    fig.add_vline(x=last_date, line_dash="dot", line_color="#80868b")
    fig.update_layout(height=450, hovermode="x unified", yaxis_title="Children per day", margin=dict(t=30), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    d7 = dsel[dsel.h == 7].iloc[0]
    t7 = tsel[tsel.h == 7].iloc[0]
    dh_ = dsel[dsel.h == horizon].iloc[0]
    th_ = tsel[tsel.h == horizon].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected discharges, next 7 days", f"{d7['point'] * 7:,.0f}", help="7 × forecast 7-day mean discharge rate at +7 days")
    k2.metric("Expected transfers into HHS, next 7 days", f"{t7['point'] * 7:,.0f}")
    net7 = (t7["point"] - d7["point"]) * 7
    k3.metric("Net intake − exits (next 7 d)", f"{net7:+,.0f}", "pressure building" if net7 > 0 else "discharges keep pace",
              delta_color="inverse")
    cover = d7["point"] / max(t7["point"], 1e-9)
    k4.metric("Discharge coverage ratio", f"{cover:.2f}×", "≥1.0 means exits offset intake", delta_color="off")
    if net7 > 0:
        st.warning("Forecast intake exceeds forecast discharges over the next 7 days — HHS care load is expected to rise absent action.")
    else:
        st.success("Forecast discharges are sufficient to offset incoming transfers over the next 7 days.")
    tbl = pd.DataFrame({"days ahead": dsel["h"].values, "date": dsel["date"].dt.date.values, "discharges/day": dsel["point"].values,
                        "transfers/day": tsel["point"].values})
    tbl["net (transfers − discharges)/day"] = tbl["transfers/day"] - tbl["discharges/day"]
    st.dataframe(tbl[tbl["days ahead"].isin([1, 3, 7, 14, 21, 28])].style.format({"discharges/day": "{:.1f}", "transfers/day": "{:.1f}", "net (transfers − discharges)/day": "{:+.1f}"}),
                 hide_index=True, width="stretch")
    st.caption("Discharge and transfer rates are modelled as trailing 7-day means, so the value at +7 days is the average daily flow over the next seven days.")

# ----------------------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Statistical vs machine-learning vs baseline models (walk-forward backtest)")
    tgt_label = st.radio("Target", ["Children in HHS care", "Discharges (7-day mean)", "Transfers into HHS (7-day mean)"], horizontal=True)
    tgt = {"Children in HHS care": "hhs_care", "Discharges (7-day mean)": "discharged", "Transfers into HHS (7-day mean)": "transferred"}[tgt_label]
    metric = st.radio("Metric", ["MAE", "RMSE", "MAPE", "Accuracy_pct"], horizontal=True)
    m = mh[mh.target == tgt]
    fig = go.Figure()
    for mo in m.model.unique():
        s = m[m.model == mo].sort_values("h")
        fig.add_trace(go.Scatter(x=s["h"], y=s[metric], name=f"{mo} ({FAMILY[mo]})", mode="lines+markers", line=dict(color=COLORS[mo], width=3 if mo == primary and tgt == "hhs_care" else 2)))
    fig.update_layout(height=420, xaxis_title="Forecast horizon (days)", yaxis_title=metric, margin=dict(t=20), xaxis=dict(tickvals=GRID))
    st.plotly_chart(fig, width="stretch")
    piv = m.pivot(index="model", columns="h", values=metric)
    piv.insert(0, "family", [FAMILY[i] for i in piv.index])
    st.markdown(f"**{metric} by horizon**")
    st.dataframe(piv.style.format({c: "{:,.1f}" for c in piv.columns if c != "family"}).highlight_min(axis=0, subset=[c for c in piv.columns if c != "family"], color="#c8e6c9") if metric != "Accuracy_pct" else
                 piv.style.format({c: "{:,.1f}" for c in piv.columns if c != "family"}).highlight_max(axis=0, subset=[c for c in piv.columns if c != "family"], color="#c8e6c9"),
                 width="stretch")
    st.markdown("**Reliability across regimes** (MAPE %, all horizons pooled): before = up to 1 Jan 2025, during = 2 Jan–15 Mar 2025, after = later")
    r = mr[mr.target == tgt].pivot(index="model", columns="regime", values="MAPE")[["pre-shift", "transition", "post-shift"]]
    st.dataframe(r.style.format("{:.1f}").highlight_min(axis=0, color="#c8e6c9"), width="stretch")
    cv = cov[cov.target == tgt].groupby("model")[["cover80", "cover95"]].mean().round(1).rename(columns={"cover80": "80% PI coverage (%)", "cover95": "95% PI coverage (%)"})
    st.markdown("**Interval calibration** (share of actual outcomes falling inside the out-of-sample prediction interval; ideal = 80 / 95)")
    st.dataframe(cv, width="stretch")

# ----------------------------------------------------------------------------- tab 4
with tab4:
    st.subheader("Scenario comparison")
    st.caption("Transparent flow-balance what-if: each scenario adds the cumulative difference between assumed and baseline daily "
               "transfers/discharges to the primary-model forecast. It is a planning aid, not a fitted causal model.")
    base_t = fc[(fc.target == "transferred") & (fc.basis == basis) & (fc.model == (primary if primary in tsel_models else "Exponential smoothing"))].sort_values("h")
    base_d = fc[(fc.target == "discharged") & (fc.basis == basis) & (fc.model == (primary if primary in tsel_models else "Exponential smoothing"))].sort_values("h")
    t_now = float(base_t["point"].iloc[:7].mean())
    d_now = float(base_d["point"].iloc[:7].mean())
    st.write(f"Baseline daily flows (forecast, next 7 days): transfers **{t_now:.1f}**/day, discharges **{d_now:.1f}**/day.")
    presets = {
        "Moderate intake surge (+50 transfers/day)": (50, 0),
        "Severe intake surge (+150 transfers/day, ~2024 levels)": (150, 0),
        "Slow placements (−50% discharges)": (0, -0.5 * d_now),
        "Combined stress (+100 transfers, −30% discharges)": (100, -0.3 * d_now),
    }
    chosen = st.multiselect("Scenarios to compare", list(presets), default=list(presets)[:3])
    c1, c2 = st.columns(2)
    cu_in = c1.slider("Custom scenario: change in transfers into HHS (children/day)", -20, 250, 25)
    cu_out = c2.slider("Custom scenario: change in discharges (children/day)", -100, 100, 0)
    scen = {"Baseline": (0.0, 0.0), **{k: presets[k] for k in chosen}, "Custom": (cu_in, cu_out)}
    sel_p = p_fc[p_fc.h <= horizon].sort_values("h")
    fig = go.Figure()
    rows = []
    palette = ["#202124", "#e8710a", "#d93025", "#1a73e8", "#8e24aa", "#00897b"]
    for i, (name, (di, do)) in enumerate(scen.items()):
        path = sel_p["point"].values + np.cumsum(np.full(len(sel_p), di + do))
        fig.add_trace(go.Scatter(x=[last_date] + list(sel_p["date"]), y=[cur] + list(path), name=name, line=dict(color=palette[i % len(palette)], width=3 if name == "Baseline" else 2)))
        breach = next((int(h) for h, v in zip(sel_p["h"], path) if v > capacity), None)
        rows.append({"Scenario": name, f"Care load at +{horizon} d": path[-1], "Change vs baseline": path[-1] - sel_p["point"].values[-1],
                     "Peak": path.max(), "First day above capacity": f"day {breach}" if breach else "—"})
    fig.add_hline(y=capacity, line_dash="dot", line_color="#d93025", annotation_text="Capacity threshold")
    fig.update_layout(height=470, hovermode="x unified", yaxis_title="Children in HHS care", margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows).style.format({f"Care load at +{horizon} d": "{:,.0f}", "Change vs baseline": "{:+,.0f}", "Peak": "{:,.0f}"}),
                 hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- tab 5
with tab5:
    st.subheader("Early-warning indicators and KPIs")
    k = kp.set_index("model")
    kk = k.loc[primary]
    a, b, c, d_ = st.columns(4)
    a.metric("Forecast accuracy (1 − WAPE, all horizons)", f"{kk['accuracy_pct']:.1f}%")
    if kk["detected"] == 0:
        b.metric("Surge lead time (mean, detected surges)", "n/a", f"no surge warnings issued ({int(kk['events'])} surges occurred)", delta_color="off")
    else:
        b.metric("Surge lead time (mean, detected surges)", f"{kk['mean_lead_days']:.1f} days", f"recall {kk['recall_pct']:.0f}% of {int(kk['events'])} surges", delta_color="off")
    c.metric("Capacity breach probability", f"{prob:.0%}", f"+{horizon} d, threshold {capacity:,}", delta_color="off")
    d_.metric("Forecast stability index", f"{kk['stability_index_pct']:.1f}%", help="100% minus the mean relative revision between forecasts for the same date issued 7 days apart")
    bd = kp.sort_values("recall_pct", ascending=False).iloc[0]
    st.info(f"Best surge detector in the backtest: **{bd['model']}** — warned ahead of {bd['recall_pct']:.0f}% of {int(bd['events'])} surges "
            f"with a mean lead time of {bd['mean_lead_days']:.0f} days, at {bd['precision_pct']:.0f}% warning precision. "
            f"Smoother statistical models rarely forecast a rise, so they miss surges; ML models warn earlier but raise more false alarms.")
    st.markdown("**Breach-probability profile over the forecast horizon**")
    probs = [breach_probability(bt, "hhs_care", primary, basis, int(h), p, capacity) for h, p in zip(p_fc["h"], p_fc["point"])]
    fig = go.Figure(go.Scatter(x=p_fc["h"], y=probs, fill="tozeroy", line=dict(color="#d93025"), name="P(care load > capacity)"))
    fig.update_layout(height=300, yaxis=dict(tickformat=".0%", range=[0, 1]), xaxis_title="Days ahead", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown("**KPI summary by model (HHS care load)**")
    show = kp.copy()
    show["family"] = show["model"].map(FAMILY)
    show = show[["model", "family", "accuracy_pct", "MAE", "RMSE", "MAPE", "recall_pct", "mean_lead_days", "precision_pct", "stability_index_pct",
                 "mape_pre_shift", "mape_transition", "mape_post_shift"]]
    show.columns = ["Model", "Family", "Accuracy %", "MAE", "RMSE", "MAPE %", "Surge recall %", "Mean lead (d)", "Warning precision %",
                    "Stability %", "MAPE pre-shift", "MAPE transition", "MAPE post-shift"]
    st.dataframe(show.style.format({c: "{:.1f}" for c in show.columns if c not in ("Model", "Family")}), hide_index=True, width="stretch")
    st.caption(f"Surge = care load rising ≥5% above the forecast-origin level within 28 days. A warning is issued when the model forecasts such a rise; "
               f"lead time is the number of days between the warning and the surge. Robustness = how MAPE holds up across the three regimes.")

# ----------------------------------------------------------------------------- tab 6
with tab6:
    st.subheader("Data and methodology")
    st.markdown(f"""
**Data.** {meta['n_observed']} reported days from {meta['first_date']} to {meta['last_date']}. Reports are published mostly Sunday–Thursday, so
{meta['n_imputed']} calendar days are missing; they were filled with time-based linear interpolation and are **excluded from all accuracy scoring**.

**Structural break.** Care load fell from ≈6,400 (Dec 2024) to ≈2,100 (Mar 2025) and has stayed in a 2,000–2,500 band since. Models are scored separately
for pre-shift, transition and post-shift periods, and prediction intervals can be based on the post-shift regime only.

**Features.** Lags (t−1, t−7, t−14), 7/14-day rolling means and variances, net pressure (transfers − discharges), CBP→HHS gap, target-date day-of-week, month and federal-holiday flag.

**Models.** Baselines (persistence, moving average) · statistical (damped-trend exponential smoothing, ARIMA(2,1,1), SARIMA(1,1,1)(1,0,1,7)) · ML (Random Forest, Gradient Boosting; direct multi-horizon models on the log-ratio of future to current level).

**Validation.** Strict time split — first test origin 2024-01-07; weekly expanding-window walk-forward origins; horizons 1, 3, 7, 14, 21, 28 days. ML models refit every 6 weeks using only data available at the origin.

**Uncertainty.** Prediction intervals are empirical quantiles of out-of-sample log forecast errors per model and horizon.
""")
    st.dataframe(data.reset_index().tail(15), hide_index=True, width="stretch")
