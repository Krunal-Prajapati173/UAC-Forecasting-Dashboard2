"""Generates the figures used in the research paper and prints EDA statistics."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from pathlib import Path
ART = Path(__file__).parent / "artifacts"; FIG = ART / "figs"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150})
d = pd.read_csv(ART/"data_clean.csv", parse_dates=["date"]).set_index("date")
dec = pd.read_csv(ART/"decomposition.csv", parse_dates=["date"]).set_index("date")
bt = pd.read_csv(ART/"backtest.csv", parse_dates=["origin","target_date"])
mh = pd.read_csv(ART/"metrics_horizon.csv"); fc = pd.read_csv(ART/"forecast_final.csv", parse_dates=["date"])
fi = pd.read_csv(ART/"feature_importance.csv")
C = {"Naive (persistence)":"#9aa0a6","Moving average (7d)":"#b39ddb","Moving average (14d)":"#b39ddb","Exponential smoothing":"#f9a825",
     "ARIMA(2,1,1)":"#26a69a","SARIMA(1,1,1)(1,0,1,7)":"#00897b","Random Forest":"#5c6bc0","Gradient Boosting":"#e53935"}
SH, SE = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-03-15")
obs = d[~d.imputed]

# Fig 1: overview of the pipeline flows
fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
ax[0].plot(d.index, d.hhs_care, color="#202124", lw=1.4); ax[0].set_ylabel("Children in HHS care")
ax[0].axvspan(SH, SE, color="#f9a825", alpha=.2); ax[0].text(SH, d.hhs_care.max()*.93, " structural\n shift", fontsize=7.5, va="top")
ax[1].plot(d.index, d.transferred.rolling(7, min_periods=1).mean(), color="#e8710a", lw=1.2, label="Transfers into HHS (7d mean)")
ax[1].plot(d.index, d.discharged.rolling(7, min_periods=1).mean(), color="#1a73e8", lw=1.2, label="Discharges (7d mean)"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_ylabel("Children / day")
ax[2].plot(d.index, d.apprehended.rolling(7, min_periods=1).mean(), color="#8e24aa", lw=1.2, label="Apprehended (7d mean)")
ax[2].plot(d.index, d.cbp_custody.rolling(7, min_periods=1).mean(), color="#00897b", lw=1.2, label="In CBP custody (7d mean)"); ax[2].legend(frameon=False, fontsize=8); ax[2].set_ylabel("Children")
fig.tight_layout(); fig.savefig(FIG/"fig1_overview.png"); plt.close()

# Fig 2: decomposition
fig, ax = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True)
ax[0].plot(d.index, d.hhs_care, color="#202124", lw=1); ax[0].set_ylabel("Observed")
ax[1].plot(dec.index, dec.hhs_care_trend, color="#5c6bc0"); ax[1].set_ylabel("Trend")
ax[2].plot(dec.index, dec.hhs_care_seasonal, color="#26a69a", lw=.8); ax[2].set_ylabel("Weekly\nseasonal")
ax[3].plot(dec.index, dec.hhs_care_resid, color="#e53935", lw=.7); ax[3].set_ylabel("Residual")
fig.tight_layout(); fig.savefig(FIG/"fig2_decomposition.png"); plt.close()

# Fig 3: day-of-week profile of flows (observed days only)
o = obs.copy(); o["dow"] = o.index.dayofweek
names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",6:"Sun"}
g = o[o.index < SH].groupby("dow")[["apprehended","transferred","discharged"]].mean()
g = g.loc[[x for x in [6,0,1,2,3,4] if x in g.index]]
fig, ax = plt.subplots(figsize=(6.4, 3)); w = .26; x = np.arange(len(g))
for i,(c,col) in enumerate([("apprehended","#8e24aa"),("transferred","#e8710a"),("discharged","#1a73e8")]):
    ax.bar(x+(i-1)*w, g[c], w, label=c.capitalize(), color=col)
ax.set_xticks(x); ax.set_xticklabels([names[i] for i in g.index]); ax.legend(frameon=False); ax.set_ylabel("Mean children / day (pre-2025)")
fig.tight_layout(); fig.savefig(FIG/"fig3_dow.png"); plt.close()

# Fig 4: error vs horizon (care load)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
for mo, s in mh[mh.target=="hhs_care"].groupby("model"):
    s = s.sort_values("h"); ax[0].plot(s.h, s.MAE, marker="o", ms=3, color=C[mo], label=mo); ax[1].plot(s.h, s.MAPE, marker="o", ms=3, color=C[mo])
ax[0].set_ylabel("MAE (children)"); ax[1].set_ylabel("MAPE (%)")
for a in ax: a.set_xlabel("Horizon (days)"); a.set_xticks([1,3,7,14,21,28])
ax[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(FIG/"fig4_error_horizon.png"); plt.close()

# Fig 5: backtest 14-day-ahead vs actual
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.plot(obs.index, obs.hhs_care, color="#202124", lw=1.3, label="Actual")
for mo in ["Exponential smoothing", "Gradient Boosting"]:
    b = bt[(bt.target=="hhs_care")&(bt.model==mo)&(bt.h==14)].sort_values("target_date")
    ax.plot(b.target_date, b.forecast, color=C[mo], lw=1.1, label=f"{mo} (14-day-ahead)")
ax.axvspan(SH, SE, color="#f9a825", alpha=.15); ax.set_ylabel("Children in HHS care"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG/"fig5_backtest14.png"); plt.close()

# Fig 6: final forecast with intervals
fig, ax = plt.subplots(figsize=(7.2, 3.4))
h = d[d.index > d.index.max()-pd.Timedelta(days=120)]; ax.plot(h.index, h.hhs_care, color="#202124", lw=1.5, label="Actual")
for mo in ["Exponential smoothing", "SARIMA(1,1,1)(1,0,1,7)", "Gradient Boosting"]:
    f = fc[(fc.target=="hhs_care")&(fc.model==mo)&(fc.basis=="recent")].sort_values("h")
    ax.plot(f.date, f.point, color=C[mo], ls="--", lw=1.4, label=mo)
f = fc[(fc.target=="hhs_care")&(fc.model=="Exponential smoothing")&(fc.basis=="recent")].sort_values("h")
ax.fill_between(f.date, f.lo95, f.hi95, color="#f9a825", alpha=.18, label="95% PI (ETS)"); ax.fill_between(f.date, f.lo80, f.hi80, color="#f9a825", alpha=.3, label="80% PI (ETS)")
ax.set_ylabel("Children in HHS care"); ax.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(FIG/"fig6_forecast.png"); plt.close()

# Fig 7: feature importance
top = fi[fi.model=="Gradient Boosting"].nlargest(10, "importance").iloc[::-1]
fig, ax = plt.subplots(figsize=(6, 3.2)); ax.barh(top.feature, top.importance, color="#e53935"); ax.set_xlabel("Importance (14-day-ahead Gradient Boosting)")
fig.tight_layout(); fig.savefig(FIG/"fig7_importance.png"); plt.close()

# Fig 8: regime MAPE
mr = pd.read_csv(ART/"metrics_regime.csv"); r = mr[mr.target=="hhs_care"].pivot(index="model", columns="regime", values="MAPE")[["pre-shift","transition","post-shift"]]
fig, ax = plt.subplots(figsize=(6.6, 3.2)); r.plot.bar(ax=ax, color=["#5c6bc0","#f9a825","#26a69a"], width=.8); ax.set_ylabel("MAPE (%)"); ax.set_xlabel("")
plt.xticks(rotation=30, ha="right", fontsize=7.5); ax.legend(frameon=False, fontsize=8, title="Regime", title_fontsize=8); fig.tight_layout(); fig.savefig(FIG/"fig8_regime.png"); plt.close()

# ---- EDA statistics
print("ADF level  p=%.3f" % adfuller(d.hhs_care)[1], " ADF diff p=%.4f" % adfuller(d.hhs_care.diff().dropna())[1])
print("ADF discharged p=%.3f" % adfuller(d.discharged)[1])
a = acf(d.hhs_care.diff().dropna(), nlags=14); print("ACF of diffs lag1,7,14:", a[1].round(2), a[7].round(2), a[14].round(2))
v = dec.hhs_care_seasonal.var()/(dec.hhs_care_seasonal.var()+dec.hhs_care_resid.var()); print("seasonal strength hhs (var ratio) %.3f"%v)
for c in ["discharged","transferred"]:
    v = dec[c+"_seasonal"].var()/(dec[c+"_seasonal"].var()+dec[c+"_resid"].var()); print(c,"seasonal strength %.3f"%v)
print(obs[["apprehended","cbp_custody","transferred","hhs_care","discharged"]].corr().round(2))
print("lagged corr transferred->apprehended:", [round(obs.transferred.corr(obs.apprehended.shift(k)),2) for k in range(0,4)])
print("flow-identity: corr(dHHS, net):", round(d.hhs_care.diff().corr(d.net_pressure),2))
for lab,(s,e) in {"pre":("2023-01-12","2024-12-31"),"post":("2025-03-16","2025-12-21")}.items():
    x=obs[s:e]; print(lab, x[["apprehended","transferred","discharged","hhs_care"]].mean().round(1).to_dict(), "disch/transf ratio", round(x.discharged.sum()/x.transferred.sum(),2))
print(obs.hhs_care.max(), obs.hhs_care.idxmax(), obs.hhs_care.min(), obs.hhs_care.idxmin())
print("net pressure mean by regime:", d.groupby("regime").net_pressure.mean().round(1).to_dict())
b=bt[(bt.target=='hhs_care')&bt.observed]; print("obs targets", b.groupby('model').size().iloc[0])
print(fi[fi.model=='Gradient Boosting'].nlargest(6,'importance').round(3).to_string())
