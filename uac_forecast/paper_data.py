import json, numpy as np, pandas as pd
from pathlib import Path
A = Path("artifacts")
mh = pd.read_csv(A/"metrics_horizon.csv"); mr = pd.read_csv(A/"metrics_regime.csv"); kp = pd.read_csv(A/"kpis_by_model.csv")
cov = pd.read_csv(A/"coverage.csv"); fc = pd.read_csv(A/"forecast_final.csv", parse_dates=["date"])
bt = pd.read_csv(A/"backtest.csv", parse_dates=["origin","target_date"]); d = pd.read_csv(A/"data_clean.csv", parse_dates=["date"]).set_index("date")
FAM = {"Naive (persistence)":"Baseline","Moving average (7d)":"Baseline","Moving average (14d)":"Baseline","Exponential smoothing":"Statistical",
       "ARIMA(2,1,1)":"Statistical","SARIMA(1,1,1)(1,0,1,7)":"Statistical","Random Forest":"ML","Gradient Boosting":"ML"}
ORDER = ["Naive (persistence)","Moving average (7d)","Moving average (14d)","Exponential smoothing","ARIMA(2,1,1)","SARIMA(1,1,1)(1,0,1,7)","Random Forest","Gradient Boosting"]
H = [1,3,7,14,21,28]
out = {}
def piv(t, metric):
    p = mh[mh.target==t].pivot(index="model", columns="h", values=metric)
    p = p.loc[[m for m in ORDER if m in p.index]]
    best = {h: p[h].idxmin() for h in H}
    return {"rows":[{"model":m,"family":FAM[m],"vals":[float(p.loc[m,h]) for h in H]} for m in p.index], "best":{str(h):best[h] for h in H}}
for t in ["hhs_care","discharged","transferred"]:
    out[f"mae_{t}"] = piv(t,"MAE"); out[f"mape_{t}"] = piv(t,"MAPE"); out[f"rmse_{t}"] = piv(t,"RMSE")
# overall
ov = {}
for t in ["hhs_care","discharged","transferred"]:
    g = mh[mh.target==t]
    ov[t] = []
import pipeline as P
for t in ["hhs_care","discharged","transferred"]:
    m = P.metric_table(bt[bt.target==t], ["model"]); m = m.set_index("model").loc[[x for x in ORDER if x in set(m.model)]]
    ov[t] = [{"model":i,"family":FAM[i],"MAE":r.MAE,"RMSE":r.RMSE,"MAPE":r.MAPE,"Acc":r.Accuracy_pct} for i,r in m.iterrows()]
out["overall"] = ov
# skill vs naive for best model each horizon (care)
sk = []
p = mh[mh.target=="hhs_care"].pivot(index="model", columns="h", values="MAE")
for h in H:
    b = p[h].idxmin(); sk.append({"h":h,"best":b,"mae":float(p.loc[b,h]),"naive":float(p.loc["Naive (persistence)",h]),"gain":float(100*(1-p.loc[b,h]/p.loc["Naive (persistence)",h]))})
out["skill_care"] = sk
pd_ = mh[mh.target=="discharged"].pivot(index="model", columns="h", values="MAE"); sk=[]
for h in H:
    b = pd_[h].idxmin(); sk.append({"h":h,"best":b,"mae":float(pd_.loc[b,h]),"naive":float(pd_.loc["Naive (persistence)",h]),"gain":float(100*(1-pd_.loc[b,h]/pd_.loc["Naive (persistence)",h]))})
out["skill_disch"] = sk
# regimes
r = mr[mr.target=="hhs_care"].pivot(index="model", columns="regime", values="MAPE")[["pre-shift","transition","post-shift"]]
rr = mr[mr.target=="hhs_care"].pivot(index="model", columns="regime", values="MAE")[["pre-shift","transition","post-shift"]]
out["regime"] = [{"model":m,"family":FAM[m],"mape":[float(x) for x in r.loc[m]],"mae":[float(x) for x in rr.loc[m]],"ratio":float(r.loc[m,"transition"]/r.loc[m,"pre-shift"])} for m in ORDER if m in r.index]
out["regime_n"] = {k:int(v) for k,v in mr[(mr.target=="hhs_care")&(mr.model=="Naive (persistence)")].set_index("regime")["n"].items()}
# KPIs
k = kp.set_index("model").loc[[m for m in ORDER if m in set(kp.model)]]
out["kpi"] = [{"model":i,"family":FAM[i],**{c:(None if pd.isna(r[c]) else float(r[c])) for c in k.columns}} for i,r in k.iterrows()]
# coverage
c = cov.groupby(["target","model"])[["cover80","cover95"]].mean().reset_index()
out["cov"] = {t:[{"model":m,"c80":float(g.loc[g.model==m,"cover80"].iloc[0]),"c95":float(g.loc[g.model==m,"cover95"].iloc[0])} for m in ORDER if m in set(g.model)] for t,g in c.groupby("target")}
# final forecasts
last = d.index.max(); cur = float(d.hhs_care.iloc[-1]); out["cur"] = cur; out["last_date"] = str(last.date())
f = fc[(fc.target=="hhs_care")&(fc.basis=="recent")]
out["fc_care"] = [{"model":m,"family":FAM[m],"vals":[float(f[(f.model==m)&(f.h==h)].point.iloc[0]) for h in [1,7,14,28]]} for m in ORDER if m in set(f.model)]
for basis in ["recent","full"]:
    f2 = fc[(fc.target=="hhs_care")&(fc.basis==basis)&(fc.model=="Exponential smoothing")]
    out[f"pi_{basis}"] = [{"h":h,"point":float(f2[f2.h==h].point.iloc[0]),"lo80":float(f2[f2.h==h].lo80.iloc[0]),"hi80":float(f2[f2.h==h].hi80.iloc[0]),"lo95":float(f2[f2.h==h].lo95.iloc[0]),"hi95":float(f2[f2.h==h].hi95.iloc[0])} for h in [1,7,14,28]]
fl = {}
for t in ["discharged","transferred"]:
    ff = fc[(fc.target==t)&(fc.basis=="recent")]
    fl[t] = [{"model":m,"vals":[float(ff[(ff.model==m)&(ff.h==h)].point.iloc[0]) for h in [1,7,14,28]]} for m in ORDER if m in set(ff.model)]
out["fc_flow"] = fl
out["flow_now"] = {"d7":float(d.discharged_7d.iloc[-1]),"t7":float(d.transferred_7d.iloc[-1])}
# breach probabilities (ETS recent basis, from backtest residuals)
def bp(model, basis, h, point, thr):
    b = bt[(bt.target=="hhs_care")&(bt.model==model)&bt.observed]
    if basis=="recent": b = b[b.origin>="2025-04-01"]
    g = min(H, key=lambda x: abs(x-h)); e = b[b.h==g]
    err = np.log((e.actual+1)/(e.forecast+1)).values
    return float((((point+1)*np.exp(err)-1) > thr).mean())
thr = [2550,2600,2700,2850]; rows=[]
for t in thr:
    row={"thr":t}
    for h in [7,14,28]:
        pt = float(fc[(fc.target=="hhs_care")&(fc.model=="Exponential smoothing")&(fc.basis=="recent")&(fc.h==h)].point.iloc[0])
        row[f"recent{h}"]=bp("Exponential smoothing","recent",h,pt,t); row[f"full{h}"]=bp("Exponential smoothing","full",h,pt,t)
    rows.append(row)
out["breach"] = rows
# misc stats
obs = d[~d.imputed]
out["stats"] = {"n_obs":int(len(obs)),"n_days":int(len(d)),"n_imp":int(d.imputed.sum()),"peak":float(obs.hhs_care.max()),"peak_date":str(obs.hhs_care.idxmax().date()),
                "min":float(obs.hhs_care.min()),"min_date":str(obs.hhs_care.idxmin().date()),
                "aug_avg":float(d.loc["2025-08"].hhs_care.mean()),"dec_avg":float(d.loc["2025-12"].hhs_care.mean())}
json.dump(out, open("artifacts/paper_data.json","w"), indent=1)
print(json.dumps(out["breach"],indent=0)); print(out["stats"]); print(out["skill_care"]); print(out["skill_disch"]); print(out["fc_flow"]); print(out["flow_now"])
print(out["regime_n"])
