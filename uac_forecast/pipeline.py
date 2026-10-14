"""
UAC Program - Predictive Forecasting of Care Load & Placement Demand
End-to-end pipeline: preparation -> features -> walk-forward backtest -> final forecasts -> KPIs.

Run:  python pipeline.py            (full run, ~10-15 min)
      python pipeline.py --quick    (fewer origins, for smoke testing)
Outputs are written to ./artifacts and read by the Streamlit app.
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
CSV = ROOT / "data" / "HHS_Unaccompanied_Alien_Children_Program.csv"

HORIZONS = [1, 3, 7, 14, 21, 28]          # backtest grid (days ahead)
ALL_H = list(range(1, 29))                # final forecast horizons
TEST_START = pd.Timestamp("2024-01-01")   # strict time split: train < 2024-01-01, walk forward after
SHIFT_START, SHIFT_END = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-03-15")
SURGE_TAU = 0.05                          # >=5% rise in care load = surge
EPS = 1.0

RENAME = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "apprehended",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transferred",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharged",
}


# --------------------------------------------------------------------------- #
# 1. Time-series preparation
# --------------------------------------------------------------------------- #
def load_and_prepare(path=CSV):
    raw = pd.read_csv(path, parse_dates=["Date"]).rename(columns=RENAME)
    raw = raw.sort_values("date").drop_duplicates("date").set_index("date")
    full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    d = raw.reindex(full_idx)
    d.index.name = "date"
    d["imputed"] = d["hhs_care"].isna()
    d = d.drop(columns=[]).copy()
    cols = ["apprehended", "cbp_custody", "transferred", "hhs_care", "discharged"]
    d[cols] = d[cols].interpolate(method="time", limit_direction="both")
    d["net_pressure"] = d["transferred"] - d["discharged"]
    d["transferred_7d"] = d["transferred"].rolling(7, min_periods=1).mean()
    d["discharged_7d"] = d["discharged"].rolling(7, min_periods=1).mean()
    d["regime"] = np.where(d.index < SHIFT_START, "pre-shift",
                   np.where(d.index <= SHIFT_END, "transition", "post-shift"))
    return raw, d


def decompose(d):
    out = pd.DataFrame(index=d.index)
    for c in ["hhs_care", "discharged", "transferred"]:
        res = STL(d[c], period=7, robust=True).fit()
        out[f"{c}_trend"], out[f"{c}_seasonal"], out[f"{c}_resid"] = res.trend, res.seasonal, res.resid
    return out


# --------------------------------------------------------------------------- #
# 2. Feature engineering (computed from information available at origin t)
# --------------------------------------------------------------------------- #
def make_features(d):
    f = pd.DataFrame(index=d.index)
    h = d["hhs_care"]
    f["hhs"] = h
    for l in (1, 7, 14):
        f[f"hhs_lag{l}"] = h.shift(l)
    f["hhs_chg7"] = h / h.shift(7) - 1
    f["hhs_chg14"] = h / h.shift(14) - 1
    for w in (7, 14):
        f[f"hhs_rmean{w}_rel"] = h.rolling(w).mean() / h - 1
        f[f"hhs_rstd{w}_rel"] = h.rolling(w).std() / h
    for c in ["apprehended", "cbp_custody", "transferred", "discharged"]:
        f[c] = d[c]
        for l in (1, 7, 14):
            f[f"{c}_lag{l}"] = d[c].shift(l)
        for w in (7, 14):
            f[f"{c}_rmean{w}"] = d[c].rolling(w).mean()
            f[f"{c}_rvar{w}"] = d[c].rolling(w).var()
    net = d["transferred"] - d["discharged"]
    f["net_pressure"] = net
    f["net_rmean7"] = net.rolling(7).mean()
    f["net_rmean14"] = net.rolling(14).mean()
    f["net_rmean7_rel"] = f["net_rmean7"] / h
    f["cbp_to_hhs_gap_rmean7"] = (d["apprehended"] - d["transferred"]).rolling(7).mean()
    f["origin_dow"] = d.index.dayofweek
    return f


_hol_cache = {}
def _holidays():
    if "h" not in _hol_cache:
        _hol_cache["h"] = set(USFederalHolidayCalendar().holidays("2022-12-01", "2027-12-31"))
    return _hol_cache["h"]


def add_target_calendar(X, h):
    tdates = X.index + pd.Timedelta(days=h)
    X = X.copy()
    X["t_dow"] = tdates.dayofweek
    X["t_month"] = tdates.month
    hol = _holidays()
    X["t_holiday"] = [int(t in hol) for t in tdates]
    return X


# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #
def ml_models():
    return {
        "Random Forest": lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_depth=10,
                                                       max_features=0.5, n_jobs=-1, random_state=0),
        "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                               subsample=0.8, random_state=0),
    }


def stat_forecasts_stock(series, max_h):
    """Statistical + baseline forecasts of the HHS care load (levels) from the history `series`."""
    out = {}
    last = float(series.iloc[-1])
    out["Naive (persistence)"] = np.full(max_h, last)
    out["Moving average (7d)"] = np.full(max_h, float(series.iloc[-7:].mean()))
    hist = series.iloc[-400:]
    try:
        m = ExponentialSmoothing(hist.values, trend="add", damped_trend=True, initialization_method="estimated").fit()
        out["Exponential smoothing"] = m.forecast(max_h)
    except Exception:
        out["Exponential smoothing"] = np.full(max_h, last)
    try:
        m = ARIMA(hist.values, order=(2, 1, 1)).fit()
        out["ARIMA(2,1,1)"] = m.forecast(max_h)
    except Exception:
        out["ARIMA(2,1,1)"] = np.full(max_h, last)
    try:
        m = SARIMAX(hist.values, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False, maxiter=60)
        out["SARIMA(1,1,1)(1,0,1,7)"] = m.forecast(max_h)
    except Exception:
        out["SARIMA(1,1,1)(1,0,1,7)"] = np.full(max_h, last)
    return {k: np.clip(np.asarray(v, float), 0, None) for k, v in out.items()}


def stat_forecasts_flow(series7, raw_series, max_h):
    """Baselines + ETS for a flow measured as a trailing 7-day mean."""
    out = {}
    last = float(series7.iloc[-1])
    out["Naive (persistence)"] = np.full(max_h, last)
    out["Moving average (14d)"] = np.full(max_h, float(raw_series.iloc[-14:].mean()))
    try:
        m = ExponentialSmoothing(series7.iloc[-400:].values, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
        out["Exponential smoothing"] = m.forecast(max_h)
    except Exception:
        out["Exponential smoothing"] = np.full(max_h, last)
    return {k: np.clip(np.asarray(v, float), 0, None) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# 4. Walk-forward backtest
# --------------------------------------------------------------------------- #
def pick_origins(d, start=TEST_START, quick=False):
    """One origin per week (Sunday, or the preceding observed day), only on days with a real observation."""
    obs = set(d.index[~d["imputed"]])
    cands = pd.date_range(start + pd.Timedelta(days=(6 - start.dayofweek) % 7), d.index.max() - pd.Timedelta(days=1), freq="7D")
    origins = []
    for c in cands:
        for k in range(0, 4):
            if c - pd.Timedelta(days=k) in obs:
                origins.append(c - pd.Timedelta(days=k)); break
    origins = sorted(set(origins))
    return origins[::4] if quick else origins


def ml_direct_backtest(target_name, y, F, d_imputed, origins, retrain_every=6):
    """Direct multi-horizon ML models on log-ratio target y[t+h]/y[t]; expanding window, refit every k origins."""
    last = y.index.max()
    recs = []
    models = ml_models()
    for h in HORIZONS:
        X = add_target_calendar(F, h)
        yy = np.log((y.shift(-h) + EPS) / (y + EPS))
        fitted = {}
        for i, o in enumerate(origins):
            tdate = o + pd.Timedelta(days=h)
            if tdate > last:
                continue
            if i % retrain_every == 0 or not fitted:
                tr_idx = X.index[X.index <= o - pd.Timedelta(days=h)]
                Xt, yt = X.loc[tr_idx], yy.loc[tr_idx]
                ok = Xt.notna().all(axis=1) & yt.notna()
                Xt, yt = Xt[ok], yt[ok]
                fitted = {n: mk().fit(Xt.values, yt.values) for n, mk in models.items()}
            xo = X.loc[[o]].values
            for n, mdl in fitted.items():
                pred = float(mdl.predict(xo)[0])
                fc = max((y.loc[o] + EPS) * np.exp(pred) - EPS, 0.0)
                recs.append((target_name, n, o, h, tdate, fc, float(y.loc[tdate]), not bool(d_imputed.loc[tdate])))
    return recs


def stat_backtest(target_name, d, origins, kind):
    last = d.index.max()
    recs = []
    for i, o in enumerate(origins):
        if kind == "stock":
            fc = stat_forecasts_stock(d["hhs_care"].loc[:o], max(HORIZONS))
            y = d["hhs_care"]
        else:
            col = target_name
            fc = stat_forecasts_flow(d[col + "_7d"].loc[:o], d[col].loc[:o], max(HORIZONS))
            y = d[col + "_7d"]
        for name, arr in fc.items():
            for h in HORIZONS:
                tdate = o + pd.Timedelta(days=h)
                if tdate > last:
                    continue
                recs.append((target_name, name, o, h, tdate, float(arr[h - 1]), float(y.loc[tdate]),
                             not bool(d["imputed"].loc[tdate])))
        if i % 10 == 0:
            print(f"   [{target_name}] stat origin {i + 1}/{len(origins)}", flush=True)
    return recs


def run_backtest(d, F, quick=False):
    origins = pick_origins(d, quick=quick)
    print(f"Walk-forward origins: {len(origins)}  ({origins[0].date()} .. {origins[-1].date()})", flush=True)
    recs = []
    t0 = time.time()
    # stock: HHS care load
    recs += stat_backtest("hhs_care", d, origins, "stock")
    print(f"  stat stock done {time.time() - t0:.0f}s", flush=True)
    recs += ml_direct_backtest("hhs_care", d["hhs_care"], F, d["imputed"], origins)
    print(f"  ml stock done {time.time() - t0:.0f}s", flush=True)
    # flows: 7-day mean discharges and transfers
    for col in ["discharged", "transferred"]:
        recs += stat_backtest(col, d, origins, "flow")
        recs += ml_direct_backtest(col, d[col + "_7d"], F, d["imputed"], origins)
        print(f"  {col} done {time.time() - t0:.0f}s", flush=True)
    bt = pd.DataFrame(recs, columns=["target", "model", "origin", "h", "target_date", "forecast", "actual", "observed"])
    bt["regime"] = np.where(bt["target_date"] < SHIFT_START, "pre-shift",
                    np.where(bt["target_date"] <= SHIFT_END, "transition", "post-shift"))
    return bt


# --------------------------------------------------------------------------- #
# 5. Metrics & KPIs
# --------------------------------------------------------------------------- #
def metric_table(bt, by):
    b = bt[bt["observed"]].copy()
    b["err"] = b["forecast"] - b["actual"]
    b["ae"] = b["err"].abs()
    b["ape"] = b["ae"] / b["actual"].clip(lower=1.0)
    g = b.groupby(by)
    res = g.agg(MAE=("ae", "mean"), RMSE=("err", lambda e: float(np.sqrt((e ** 2).mean()))),
                MAPE=("ape", lambda e: 100 * e.mean()), n=("ae", "size"),
                abs_sum=("ae", "sum"), act_sum=("actual", "sum")).reset_index()
    res["Accuracy_pct"] = 100 * (1 - res["abs_sum"] / res["act_sum"])   # 1 - WAPE
    return res.drop(columns=["abs_sum", "act_sum"])


def residual_quantiles(bt, target, model, basis, qs=(0.025, 0.10, 0.90, 0.975)):
    b = bt[(bt.target == target) & (bt.model == model) & bt.observed]
    if basis == "recent":
        b = b[b.origin >= pd.Timestamp("2025-04-01")]
    out = {}
    for h in HORIZONS:
        e = np.log((b[b.h == h]["actual"] + EPS) / (b[b.h == h]["forecast"] + EPS)).values
        if len(e) < 8:
            e = np.log((b["actual"] + EPS) / (b["forecast"] + EPS)).values
        out[h] = np.quantile(e, qs)
    return out


def interp_q(qdict, h):
    hs = sorted(qdict)
    arr = np.array([qdict[k] for k in hs])
    return np.array([np.interp(h, hs, arr[:, j]) for j in range(arr.shape[1])])


def coverage_table(bt):
    """Out-of-sample PI calibration: quantiles come only from origins whose targets were known at forecast time."""
    rows = []
    b = bt[bt.observed].copy()
    for (tg, mo, h), g in b.groupby(["target", "model", "h"]):
        g = g.sort_values("origin")
        e = np.log((g["actual"] + EPS) / (g["forecast"] + EPS)).values
        tdates, origs = g["target_date"].values, g["origin"].values
        hit80 = hit95 = n = 0
        for i in range(len(g)):
            known = e[tdates < origs[i]]
            if len(known) < 10:
                continue
            q = np.quantile(known, [0.025, 0.10, 0.90, 0.975])
            n += 1
            hit80 += q[1] <= e[i] <= q[2]
            hit95 += q[0] <= e[i] <= q[3]
        if n:
            rows.append((tg, mo, h, n, 100 * hit80 / n, 100 * hit95 / n))
    return pd.DataFrame(rows, columns=["target", "model", "h", "n", "cover80", "cover95"])


def surge_kpis(bt, d, model, tau=SURGE_TAU):
    """Surge lead time: for origins followed by a real >=tau rise in care load within 28 days, did the
    model warn (forecast >= tau) and how many days ahead of the surge was the warning?"""
    b = bt[(bt.target == "hhs_care") & (bt.model == model)].copy()
    b["lvl"] = d["hhs_care"].reindex(b["origin"]).values
    b["f_chg"] = b["forecast"] / b["lvl"] - 1
    b["a_chg"] = b["actual"] / b["lvl"] - 1
    events, false_alarms, warned_leads, n_warn = [], 0, [], 0
    for o, g in b.groupby("origin"):
        g = g.sort_values("h")
        real = g[(g["a_chg"] >= tau) & g["observed"]]
        warn = (g["f_chg"] >= tau).any()
        n_warn += int(warn)
        if len(real):
            first_h = int(real["h"].iloc[0])
            events.append(first_h)
            if warn:
                warned_leads.append(first_h)
        elif warn:
            false_alarms += 1
    n_ev = len(events)
    return {
        "events": n_ev,
        "detected": len(warned_leads),
        "recall_pct": 100 * len(warned_leads) / n_ev if n_ev else np.nan,
        "mean_lead_days": float(np.mean(warned_leads)) if warned_leads else 0.0,
        "median_lead_days": float(np.median(warned_leads)) if warned_leads else 0.0,
        "warnings_issued": n_warn,
        "false_alarms": false_alarms,
        "precision_pct": 100 * len(warned_leads) / n_warn if n_warn else np.nan,
    }


def stability_index(bt, target, model):
    """1 - mean relative revision between forecasts for the same target date issued 7 days apart."""
    b = bt[(bt.target == target) & (bt.model == model) & (bt.h.isin([7, 14, 21, 28]))]
    a = b.rename(columns={"forecast": "f_long", "h": "h_long"})[["target_date", "h_long", "f_long", "actual"]]
    s = b.rename(columns={"forecast": "f_short", "h": "h_short"})[["target_date", "h_short", "f_short"]]
    m = a.merge(s, on="target_date")
    m = m[m["h_long"] - m["h_short"] == 7]
    if m.empty:
        return np.nan
    rev = (m["f_long"] - m["f_short"]).abs() / m["actual"].clip(lower=1)
    return float(100 * (1 - rev.mean()))


# --------------------------------------------------------------------------- #
# 6. Final forecast from the latest observation
# --------------------------------------------------------------------------- #
def final_forecasts(d, F, bt):
    last = d.index.max()
    dates = [last + pd.Timedelta(days=h) for h in ALL_H]
    rows = []
    fits = {}
    for target, y, kind in [("hhs_care", d["hhs_care"], "stock"),
                            ("discharged", d["discharged_7d"], "flow"),
                            ("transferred", d["transferred_7d"], "flow")]:
        if kind == "stock":
            pts = stat_forecasts_stock(d["hhs_care"], max(ALL_H))
        else:
            col = target
            pts = stat_forecasts_flow(d[col + "_7d"], d[col], max(ALL_H))
        # ML direct models for every horizon 1..28
        ml = {n: [] for n in ml_models()}
        for h in ALL_H:
            X = add_target_calendar(F, h)
            yy = np.log((y.shift(-h) + EPS) / (y + EPS))
            tr = X.index[X.index <= last - pd.Timedelta(days=h)]
            Xt, yt = X.loc[tr], yy.loc[tr]
            ok = Xt.notna().all(axis=1) & yt.notna()
            for n, mk in ml_models().items():
                m = mk().fit(Xt[ok].values, yt[ok].values)
                ml[n].append(max((y.loc[last] + EPS) * np.exp(float(m.predict(X.loc[[last]].values)[0])) - EPS, 0.0))
                if h == 14 and target == "hhs_care":
                    fits[n] = (m, list(X.columns), Xt[ok])
        for n, v in ml.items():
            pts[n] = np.array(v)
        for model, arr in pts.items():
            for basis in ("full", "recent"):
                q = residual_quantiles(bt, target, model, basis) if model in set(bt[bt.target == target].model) else None
                for h, dt in zip(ALL_H, dates):
                    p = float(arr[h - 1])
                    lo95, lo80, hi80, hi95 = (np.nan,) * 4
                    if q is not None:
                        e = interp_q(q, h)   # [2.5, 10, 90, 97.5] log-ratio quantiles
                        lo95, lo80, hi80, hi95 = [max((p + EPS) * np.exp(x) - EPS, 0.0) for x in e]
                    rows.append((target, model, basis, h, dt, p, lo80, hi80, lo95, hi95))
    fc = pd.DataFrame(rows, columns=["target", "model", "basis", "h", "date", "point", "lo80", "hi80", "lo95", "hi95"])
    return fc, fits


def feature_importance(fits):
    rows = []
    for n, (m, cols, Xt) in fits.items():
        imp = m.feature_importances_
        for c, v in zip(cols, imp):
            rows.append((n, c, float(v)))
    return pd.DataFrame(rows, columns=["model", "feature", "importance"])


# --------------------------------------------------------------------------- #
def main(quick=False):
    t0 = time.time()
    raw, d = load_and_prepare()
    F = make_features(d)
    dec = decompose(d)
    d.to_csv(ART / "data_clean.csv")
    dec.to_csv(ART / "decomposition.csv")
    print(f"Prepared {len(raw)} observed days -> {len(d)} daily rows ({int(d.imputed.sum())} imputed)", flush=True)

    bt = run_backtest(d, F, quick=quick)
    bt.to_csv(ART / "backtest.csv", index=False)

    mh = metric_table(bt, ["target", "model", "h"])
    mr = metric_table(bt, ["target", "model", "regime"])
    mh.to_csv(ART / "metrics_horizon.csv", index=False)
    mr.to_csv(ART / "metrics_regime.csv", index=False)
    mr_h = metric_table(bt, ["target", "model", "regime", "h"])
    mr_h.to_csv(ART / "metrics_regime_horizon.csv", index=False)
    cov = coverage_table(bt)
    cov.to_csv(ART / "coverage.csv", index=False)
    print(f"Backtest metrics written ({time.time() - t0:.0f}s)", flush=True)

    # KPI summary per model for HHS care load
    kp = []
    for mo in sorted(bt[bt.target == "hhs_care"].model.unique()):
        s = surge_kpis(bt, d, mo)
        ms = metric_table(bt[(bt.target == "hhs_care") & (bt.model == mo)], ["model"])
        post = mr[(mr.target == "hhs_care") & (mr.model == mo)].set_index("regime")["MAPE"]
        pre, ps = post.get("pre-shift", np.nan), post.get("post-shift", np.nan)
        kp.append({"model": mo, "accuracy_pct": float(ms["Accuracy_pct"].iloc[0]), "MAE": float(ms["MAE"].iloc[0]),
                   "RMSE": float(ms["RMSE"].iloc[0]), "MAPE": float(ms["MAPE"].iloc[0]),
                   "stability_index_pct": stability_index(bt, "hhs_care", mo),
                   "mape_pre_shift": float(pre), "mape_transition": float(post.get("transition", np.nan)),
                   "mape_post_shift": float(ps), **s})
    pd.DataFrame(kp).to_csv(ART / "kpis_by_model.csv", index=False)

    fc, fits = final_forecasts(d, F, bt)
    fc.to_csv(ART / "forecast_final.csv", index=False)
    feature_importance(fits).to_csv(ART / "feature_importance.csv", index=False)
    meta = {"last_date": str(d.index.max().date()), "first_date": str(d.index.min().date()),
            "n_observed": int(len(raw)), "n_imputed": int(d.imputed.sum()),
            "n_origins": int(bt.origin.nunique()), "run_seconds": round(time.time() - t0)}
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", meta, flush=True)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
