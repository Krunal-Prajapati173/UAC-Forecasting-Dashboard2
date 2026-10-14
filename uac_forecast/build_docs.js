const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel, AlignmentType,
  LevelFormat, WidthType, ShadingType, BorderStyle, PageBreak, Footer, PageNumber,
} = require("docx");

const D = JSON.parse(fs.readFileSync("artifacts/paper_data.json", "utf8"));
const FIG = "artifacts/figs/";
const FIGDIM = { "fig1_overview": [7.2, 6.6], "fig2_decomposition": [7.2, 6.4], "fig3_dow": [6.4, 3], "fig4_error_horizon": [7.4, 3.2],
  "fig5_backtest14": [7.2, 3.3], "fig6_forecast": [7.2, 3.4], "fig7_importance": [6, 3.2], "fig8_regime": [6.6, 3.2], "fig9_exec": [7.2, 3.2] };
const FONT = "Arial";
const W = 9360;
const f0 = (x) => Math.round(x).toLocaleString("en-US");
const f1 = (x) => x.toFixed(1);
const pct = (x) => (100 * x).toFixed(0) + "%";

function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }) : new TextRun({ text: t, font: FONT, ...base }));
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, { size: o.size || 21, color: o.color, italics: o.italics }), spacing: { after: o.after ?? 120, line: 276 }, alignment: o.align, keepNext: o.keepNext });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT, bold: true, size: 30, color: "1F3864" })], spacing: { before: 280, after: 120 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT, bold: true, size: 24, color: "2F5496" })], spacing: { before: 200, after: 100 } });
const B = (t, o = {}) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(t, { size: o.size || 21 }), spacing: { after: 70, line: 264 } });
const N = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(t, { size: 21 }), spacing: { after: 80, line: 264 } });
const CAP = (t) => new Paragraph({ children: runs(t, { size: 18, italics: true, color: "555555" }), spacing: { before: 40, after: 160 }, alignment: AlignmentType.CENTER });
function IMG(name, widthIn, cap) {
  const [w, h] = FIGDIM[name]; const px = widthIn * 96;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 0 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + name + ".png"), transformation: { width: px, height: px * h / w },
      altText: { title: cap, description: cap, name } })] }), CAP(cap)];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, o = {}) {
  return new TableCell({ width: { size: w, type: WidthType.DXA }, borders, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ alignment: o.align || AlignmentType.LEFT, children: runs(String(text), { size: o.size || 17, bold: o.bold, color: o.color }) })] });
}
// rows: array of arrays; widths sum to W; highlight: Set of "r,c" for best cells
function TBL(headers, rows, widths, hl = new Set(), firstColLeft = true) {
  const total = widths.reduce((a, b) => a + b, 0);
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { fill: "1F3864", bold: true, color: "FFFFFF", align: i === 0 && firstColLeft ? AlignmentType.LEFT : AlignmentType.CENTER })) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((c, ci) => cell(c, widths[ci], {
    align: ci === 0 && firstColLeft ? AlignmentType.LEFT : AlignmentType.CENTER, fill: hl.has(ri + "," + ci) ? "C6E0B4" : (ri % 2 ? "F2F2F2" : undefined), bold: hl.has(ri + "," + ci) })) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
const SP = () => new Paragraph({ spacing: { after: 120 }, children: [] });
const short = (m) => m.replace("SARIMA(1,1,1)(1,0,1,7)", "SARIMA").replace("Naive (persistence)", "Naive").replace("Moving average", "Moving avg").replace("Exponential smoothing", "Exp. smoothing").replace("Gradient Boosting", "Grad. Boosting");

// ---- table builders
const HZ = [1, 3, 7, 14, 21, 28];
function horizonTable(key, dec = 1) {
  const t = D[key]; const rows = t.rows.map((r) => [short(r.model), r.family, ...r.vals.map((v) => v.toFixed(dec))]);
  const hl = new Set(); HZ.forEach((h, ci) => { const col = t.rows.map((r) => r.vals[ci]); const ri = col.indexOf(Math.min(...col)); hl.add(ri + "," + (ci + 2)); });
  return TBL(["Model", "Family", ...HZ.map((h) => h + "d")], rows, [2000, 1360, ...Array(6).fill(1000)], hl);
}

// ================================================================== PAPER
const care = D.overall.hhs_care;
const ets = care.find((m) => m.model === "Exponential smoothing");
const kpi = (m) => D.kpi.find((k) => k.model === m);
const gb = kpi("Gradient Boosting"), rf = kpi("Random Forest");
const pi = D.pi_recent;
const pr = (h) => pi.find((x) => x.h === h);

const paper = [];
paper.push(new Paragraph({ spacing: { before: 1200, after: 120 }, children: [new TextRun({ text: "Forecasting Care Load and Placement Demand in the UAC Program", font: FONT, bold: true, size: 44, color: "1F3864" })] }));
paper.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "A time-series and machine-learning study of HHS daily data, January 2023 to December 2025", font: FONT, size: 26, color: "595959" })] }));
paper.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "2F5496", space: 4 } }, spacing: { after: 240 }, children: [] }));
paper.push(H2("Abstract"));
paper.push(P(`We used the daily UAC Program figures (${D.stats.n_obs} reported days from 12 January 2023 to 21 December 2025) to forecast three things: how many children will be in HHS care, how many will be discharged each day, and how many will be transferred in. Baseline, statistical and machine-learning models were compared with walk-forward testing on 101 weekly forecast dates and horizons from 1 to 28 days. The biggest event in the data is a sharp drop in care load, from about 6,400 children in December 2024 to about 2,100 by March 2025, and it drives most of the forecast error. Over the whole test period, exponential smoothing and SARIMA predicted care load best (about 95% accurate), and both beat a simple "no change" forecast by 12 to 39% depending on the horizon. Random Forest and Gradient Boosting did better for discharges at one to four weeks and were the only models that gave useful advance warning of surges, though they raised many false alarms. At the last reported date there were ${f0(D.cur)} children in care, and the statistical forecast is flat at about 2,500 for the next four weeks.`));

paper.push(H1("1. Introduction"));
paper.push(P("Border activity, enforcement policy and humanitarian events can change the number of children entering federal care within a few days. Reports on past activity tell planners what already happened. What they need is an idea of how many children will be in HHS care over the next few weeks, whether discharges will keep up with new transfers, and when to add shelter space, clinical staff and caseworkers."));
paper.push(P("This study has three main goals: forecast the number of children in HHS care, estimate the gap between incoming transfers and discharges, and predict short-term discharge demand. We also wanted to see how reliable the forecasts are, whether they can warn of a surge early, and whether machine-learning models are worth using instead of classical statistical ones. A Streamlit dashboard and a short executive summary go with this paper."));

paper.push(H1("2. Data and preparation"));
paper.push(P("The dataset has five daily series: children apprehended and placed in CBP custody, children in CBP custody, children transferred out of CBP custody (the flow into HHS), children in HHS care, and children discharged from HHS care to sponsors."));
paper.push(P(`Reports come out mostly Sunday to Thursday. The ${D.stats.n_obs} reported days cover ${D.stats.n_days} calendar days, so ${D.stats.n_imp} days are missing. Most are Fridays and Saturdays, a few are holidays, and there is one 10-day gap in January 2023. We built a continuous daily calendar and filled the missing days by linear interpolation, flagging every filled day. Filled days are used as model inputs but never as targets when we measure accuracy. The long gap in January 2023 also makes the first weeks of the flow series look like a ramp-up, so those weeks should be read with some care.`));
paper.push(...IMG("fig1_overview", 5.9, "Figure 1. Daily series (flows shown as 7-day means). The shaded band marks the early-2025 drop."));

paper.push(H1("3. What the data show"));
paper.push(H2("3.1 Level and the early-2025 break"));
paper.push(P(`Care load peaked at ${f0(D.stats.peak)} on ${D.stats.peak_date}. It rose through 2023, fell through 2024 (a monthly average of about 8,850 in January and about 5,900 in November), and then dropped between 2 January and 15 March 2025 to roughly 2,100. Since then it has stayed between ${f0(D.stats.min)} (on ${D.stats.min_date}) and about 2,500. The monthly average rose from about ${f0(D.stats.aug_avg)} in August 2025 to about ${f0(D.stats.dec_avg)} in December.`));
paper.push(P("The flows fell by more than ten times over the same weeks. Before 2025 the average day had about 134 apprehensions, 184 transfers into HHS and 245 discharges. Since mid-March 2025 the averages are about 8, 10 and 12. Discharges were higher than transfers in both periods (a ratio of about 1.33 before and 1.22 after), so that ratio alone does not tell us whether the care load is rising or falling. Care load can also change through routes this dataset does not record."));
paper.push(H2("3.2 Persistence and weekly pattern"));
paper.push(P("Care load is not stationary in levels (Augmented Dickey-Fuller p = 0.66) but becomes stationary after one round of differencing (p < 0.001), so we allowed for one difference in the ARIMA models. Day-to-day changes are strongly autocorrelated: 0.59 at lag 1, 0.62 at lag 7 and 0.47 at lag 14. That points to momentum plus a weekly rhythm. An STL decomposition with a 7-day period gives a seasonal share of about 0.41 for care load, 0.66 for discharges and 0.22 for transfers, where the share is seasonal variance divided by seasonal plus residual variance. Part of the care-load figure comes from interpolating the unreported Fridays and Saturdays, so the discharge figure is the more trustworthy one."));
paper.push(...IMG("fig2_decomposition", 5.6, "Figure 2. STL decomposition of children in HHS care (7-day period)."));
paper.push(...IMG("fig3_dow", 4.6, "Figure 3. Average daily flows by day of the week, before 2025. Discharges are highest on Thursdays and Sundays."));
paper.push(H2("3.3 How the series relate"));
paper.push(P("The intake series move together. Apprehensions and transfers into HHS correlate at 0.89, and the link is strongest (0.92) when transfers are compared with apprehensions from the day before, which fits a short processing lag at CBP. Care load is most closely tied to discharges (0.92 in levels) and to transfers (0.71). The daily change in care load, however, correlates only 0.35 with transfers minus discharges. Net pressure is a useful early signal but it is not an exact accounting identity, probably because of inflows we cannot see, reporting timing and the interpolated days. This matters for the scenario analysis in Section 6."));

paper.push(H1("4. Methods"));
paper.push(H2("4.1 Features"));
paper.push(P("Every feature uses only information available on the forecast date. We used the current value of each series and its values 1, 7 and 14 days earlier, plus rolling 7-day and 14-day means and variances. We added the 7- and 14-day change in care load, net pressure (transfers minus discharges) with its rolling means, and the gap between apprehensions and transfers. For the day being forecast we also included the day of the week, the month and a flag for US federal holidays."));
paper.push(H2("4.2 Models"));
paper.push(P("The baselines are naive persistence (tomorrow equals today) and a moving average, 7 days for care load and 14 days for flows. The statistical models are damped-trend exponential smoothing, ARIMA(2,1,1) and SARIMA(1,1,1)(1,0,1,7), each refit at every forecast date on the latest 400 days. The machine-learning models are a Random Forest and a Gradient Boosting regressor with 100 trees each. We trained one model per horizon, predicting the log of the ratio between the future value and the current one, which keeps the predictions sensible across the very different scales before and after 2025. The ML models were refit every six forecast dates using only targets that were already known."));
paper.push(P("Discharges and transfers are modelled as 7-day trailing means, so the forecast at +7 days is the average daily flow over the following week. Only exponential smoothing was used as a statistical model for the flows."));
paper.push(H2("4.3 Validation"));
paper.push(P("The split is strictly by time. Models are trained on data before 1 January 2024, then tested on 101 weekly forecast dates from 7 January 2024 to 14 December 2025 with an expanding training window. We scored horizons of 1, 3, 7, 14, 21 and 28 days, only on days with a real observation, which gave 523 scored care-load forecasts per model. We also report results by period, using the date of the target: before 2 January 2025 (262 forecasts), 2 January to 15 March 2025 (55), and after 15 March 2025 (206)."));
paper.push(H2("4.4 Metrics"));
paper.push(P("Accuracy is measured with MAE, RMSE and MAPE at each horizon. We also report accuracy as 100 × (1 − total absolute error ÷ total actual), which is steadier than MAPE when counts are small. Prediction intervals come from the quantiles of past out-of-sample log errors for each model and horizon, using either the full history or only the period after the 2025 drop. To check calibration we used only errors that were known at each forecast date."));
paper.push(P("Capacity breach probability is the share of the simulated forecast distribution above a threshold the user chooses. A surge is a rise of at least 5% above the starting level within 28 days. A model gives a warning if it forecasts such a rise, and lead time is the number of days between the warning and the surge. The stability index is 100% minus the average relative change between forecasts for the same date made seven days apart. For robustness we compare each model's error before, during and after the 2025 drop."));

paper.push(H1("5. Results"));
paper.push(H2("5.1 Care load by horizon"));
paper.push(P("Table 1 gives the mean absolute error, in children, over the whole test period. The best value at each horizon is shaded.", { keepNext: true }));
paper.push(horizonTable("mae_hhs_care", 1));
paper.push(CAP("Table 1. Care load: MAE (children) by horizon, walk-forward test, 2024 to 2025."));
paper.push(P(`SARIMA is the most accurate model up to seven days, and exponential smoothing is best from 14 to 28 days. Compared with naive persistence, the best model lowers the error by ${D.skill_care.map((s) => f0(s.gain) + "% at " + s.h + " days").join(", ")}. Persistence is hard to beat at one day because care load changes slowly. Random Forest and Gradient Boosting beat persistence up to about a week, but from 14 days on they are worse than it. At 28 days their MAE is 624 and 668, compared with 503 for persistence.`));
paper.push(...IMG("fig4_error_horizon", 5.9, "Figure 4. Care-load error by horizon (MAE and MAPE)."));
paper.push(TBL(["Model", "Family", "MAE", "RMSE", "MAPE %", "Accuracy %"],
  care.map((m) => [short(m.model), m.family, f1(m.MAE), f1(m.RMSE), f1(m.MAPE), f1(m.Acc)]), [2400, 1500, 1200, 1200, 1500, 1560],
  new Set([`${care.indexOf(ets)},2`, `${care.indexOf(ets)},4`, `${care.indexOf(ets)},5`, `${care.indexOf(ets)},3`])));
paper.push(CAP("Table 2. Care load, all horizons combined. Exponential smoothing and SARIMA tie for best accuracy."));
paper.push(H2("5.2 Before, during and after the 2025 drop"));
paper.push(TBL(["Model", "Before MAPE %", "During MAPE %", "After MAPE %", "During ÷ before"],
  D.regime.map((r) => [short(r.model), f1(r.mape[0]), f1(r.mape[1]), f1(r.mape[2]), f1(r.ratio) + "×"]), [2400, 1700, 1700, 1700, 1860],
  (() => { const hl = new Set(); [0, 1, 2].forEach((c) => { const col = D.regime.map((r) => r.mape[c]); hl.add(col.indexOf(Math.min(...col)) + "," + (c + 1)); }); const rc = D.regime.map((r) => r.ratio); hl.add(rc.indexOf(Math.min(...rc)) + ",4"); return hl; })()));
paper.push(CAP("Table 3. Care-load MAPE by period, all horizons combined."));
paper.push(P("Every model had trouble during the drop in early 2025, when care load fell about 65% (from a December 2024 average near 6,400 to about 2,200 by mid-March). Errors were three to five times larger than before. Exponential smoothing was the steadiest overall, at 4.1%, 13.7% and 2.3% across the three periods. Random Forest lost the least accuracy relative to its normal level, while the moving average and persistence lost the most. After the drop, errors are small for every model (2.3 to 4.0%) because care load is stable. Counts are low now, though, so small absolute errors still show up as larger percentages."));
paper.push(...IMG("fig8_regime", 5.2, "Figure 5. Care-load MAPE by period."));
paper.push(...IMG("fig5_backtest14", 5.9, "Figure 6. 14-day-ahead forecasts against actual care load. Both models trail the January to March 2025 decline."));
paper.push(H2("5.3 Discharges and the balance with transfers"));
paper.push(P("Table 4 shows MAE for the 7-day average daily discharge rate. Most of the absolute error comes from the high-volume period before 2025.", { keepNext: true }));
paper.push(horizonTable("mae_discharged", 1));
paper.push(CAP("Table 4. Discharge demand (7-day mean, children per day): MAE by horizon."));
paper.push(P(`The machine-learning models forecast discharges best from seven days on. Gradient Boosting is best at 1 and 7 days and Random Forest at 14, 21 and 28 days, lowering error against persistence by ${D.skill_disch.filter((s) => s.h >= 14).map((s) => f0(s.gain) + "% at " + s.h + " days").join(", ")}. At three days persistence is as good as anything. Discharges follow a weekday pattern and depend on recent variability and the flow out of CBP, which tree models can pick up and the smoother statistical models cannot. Transfers were harder to forecast: beyond a week, persistence and the 14-day moving average were about as good as the other models.`));
paper.push(P(`In the latest week discharges averaged ${f1(D.flow_now.d7)} children a day and transfers into HHS ${f1(D.flow_now.t7)}. For the coming week the Gradient Boosting and Random Forest models expect about 11.8 to 12.1 discharges a day against about 9.2 to 9.3 transfers. That is roughly 80 to 85 discharges against about 65 transfers, so discharges should keep up with incoming children, with a net outflow of around 20 over the week. Because net pressure tracks daily care-load changes only loosely (Section 3.3), this should be read as a direction and not an exact count.`));
paper.push(H2("5.4 Uncertainty"));
paper.push(TBL(["Model", "80% interval coverage", "95% interval coverage"],
  D.cov.hhs_care.map((c) => [short(c.model), f1(c.c80) + "%", f1(c.c95) + "%"]), [3600, 2880, 2880]));
paper.push(CAP("Table 5. How often actual care load fell inside the prediction interval (target 80% and 95%)."));
paper.push(P("The 80% intervals held up well, containing 76.5 to 81.2% of the actual values. The 95% intervals were a little too narrow, at 85.0 to 90.5%, because the 2025 drop produced some very large errors. Intervals based only on the period after the drop are much tighter than those based on the full history. For exponential smoothing at 28 days, the 80% interval is 2,338 to 2,670 on the post-drop basis and 2,097 to 2,740 on the full-history basis (for 95%: 2,227 to 2,689 against 1,640 to 2,886). We suggest planners use the post-drop basis while conditions stay stable and switch to the full-history basis if they suspect another shift."));
paper.push(H2("5.5 Early warning, stability and accuracy"));
const kr = D.kpi.map((k) => [short(k.model), f1(k.accuracy_pct), k.detected === 0 ? "none" : f1(k.mean_lead_days), k.recall_pct == null ? "-" : f0(k.recall_pct) + "%", k.precision_pct == null ? "-" : f0(k.precision_pct) + "%", f1(k.stability_index_pct)]);
paper.push(TBL(["Model", "Accuracy %", "Mean lead time (days)", "Surges caught", "Warning precision", "Stability %"], kr, [2200, 1300, 1700, 1400, 1500, 1260]));
paper.push(CAP("Table 6. Care-load indicators. There were 26 surges (a rise of 5% or more within 28 days) across the 101 forecast dates."));
paper.push(P(`The smoother statistical models almost never forecast a rise, so they gave few or no warnings. Gradient Boosting warned ahead of ${gb.detected} of ${gb.events} surges (${f0(gb.recall_pct)}%) with a mean lead time of ${f1(gb.mean_lead_days)} days. Random Forest caught ${rf.detected} (${f0(rf.recall_pct)}%) with ${f1(rf.mean_lead_days)} days of lead time. Precision was low, about 31 to 32%, so roughly two of every three warnings were false alarms. We would use these models to trigger closer monitoring, not to trigger a scale-up by themselves.`));
paper.push(P("Stability was between 93% and 96% for all models, which means the forecast for a given date moved by only 4 to 7% on average from one weekly update to the next. The ML models were the least stable (92.9% and 93.2%). Breach probability depends on the capacity threshold planners choose, which is covered in Section 6."));
paper.push(H2("5.6 What the ML models rely on"));
paper.push(...IMG("fig7_importance", 4.7, "Figure 7. Feature importance for the 14-day-ahead Gradient Boosting model."));
paper.push(P("Recent momentum in care load (the 7- and 14-day change and the distance from the 7-day mean) matters most, followed by how variable discharges have been and by CBP custody a couple of weeks earlier. The month of the target date ranks first. With only three years of data it is probably standing in for when the 2025 drop happened, and for seasonal intake, so we would not read much into it until there is a longer history."));
paper.push(H2("5.7 Forecast from 21 December 2025"));
paper.push(TBL(["Model", "Family", "+1 day", "+7 days", "+14 days", "+28 days"], D.fc_care.map((m) => [short(m.model), m.family, ...m.vals.map(f0)]), [2400, 1500, 1365, 1365, 1365, 1365]));
paper.push(CAP(`Table 7. Forecast care load by model (latest reported value: ${f0(D.cur)}).`));
paper.push(P("The statistical and baseline models agree on a flat outlook of about 2,480 to 2,550 children through mid-January 2026. Random Forest and Gradient Boosting instead forecast a fall to about 1,700 to 1,750 by day 28. We do not recommend planning around that. These models were less accurate than the statistical ones at long horizons, and the current movement is a slow rise from the August low, not a decline. Their forecast most likely reflects the 2025 drop in the training data. It is still a useful reminder of how much the long-range outlook depends on the model chosen."));
paper.push(...IMG("fig6_forecast", 5.9, "Figure 8. Care-load forecasts. The shaded bands are the 80% and 95% intervals for exponential smoothing (post-drop basis). The steep red Gradient Boosting line is the outlier discussed above."));

paper.push(H1("6. Early warning and scenarios"));
paper.push(P("In the dashboard, planners can set a capacity threshold equal to their real bed capacity. Table 8 shows breach probabilities for exponential smoothing at a few example thresholds, given as a percentage above today's care load."));
paper.push(TBL(["Threshold (children)", "vs today", "+7d (post-drop)", "+14d (post-drop)", "+28d (post-drop)", "+28d (full history)"],
  D.breach.map((b) => [f0(b.thr), "+" + (100 * (b.thr / D.cur - 1)).toFixed(1) + "%", pct(b.recent7), pct(b.recent14), pct(b.recent28), pct(b.full28)]), [1700, 1200, 1600, 1600, 1600, 1660]));
paper.push(CAP("Table 8. Capacity breach probability (exponential smoothing). The thresholds are examples only."));
paper.push(P("At a threshold 15% above today's load (2,850) the chance of a breach within four weeks is close to zero. Risk only shows up for thresholds within a few percent of the current level, and it grows with the horizon as the interval widens."));
paper.push(P("The scenario view adds up the daily difference between assumed and forecast transfers and discharges and adds it to the baseline forecast. It is a simple what-if, not a causal model. If transfers rose by 150 children a day, close to 2024 volumes, with no change in discharges, care load would be about 2,100 higher than baseline after 14 days. A rise of 50 a day would add about 700. Since daily care-load changes follow net pressure only loosely, these numbers are upper-end planning figures."));

paper.push(H1("7. Recommendations"));
paper.push(N("Plan on the statistical forecasts: SARIMA for the coming week and exponential smoothing for two to four weeks ahead. The current outlook is about 2,500 children, with an 80% range of 2,340 to 2,670 at four weeks. Nothing in the forecasts calls for a scale-up, but standby arrangements are sensible since care load has risen about 22% since August 2025."));
paper.push(N("Use Random Forest and Gradient Boosting for discharge planning at one to four weeks, and treat their warnings of a rise in care load as a prompt to watch more closely. Because about two in three warnings are false, escalate only if a warning repeats across consecutive weekly forecasts or if transfers exceed discharges for several days. This rule is our suggestion and has not been tested."));
paper.push(N("Enter the real operational capacity in the dashboard so that breach probabilities mean something, and review them every week."));
paper.push(N("Keep an eye on forecast error. During the 2025 transition it was three to five times its usual size. If the rolling error rises well above the current 2 to 4% MAPE, switch to the full-history intervals and widen planning ranges."));
paper.push(N("Improve the data by reporting on Fridays and Saturdays, adding other routes into HHS care, and adding indicators for policy changes and border activity. This would cut the noise from interpolation and might help anticipate shifts."));
paper.push(N("Refit the models at least every six weeks, and straight away after any sustained shift."));

paper.push(H1("8. Limitations"));
paper.push(P("The data contain one major structural break, and after January 2024 the test period is mostly decline and stability. The models are therefore not well tested on surges like those of 2023 and early 2024. Missing days were interpolated, which affects the weekly pattern and the daily flow forecasts. The surge definition and capacity thresholds are our assumptions and should be adapted. Surge counts are per forecast date, so they are not independent events. We did not tune the ML models, and every interval comes from pooled historical errors, not from a model-specific distribution. Flows since 2025 are only about 10 children a day, which makes percentage errors jumpy. Finally, the scenario figures are arithmetic and not estimates of cause and effect."));

paper.push(H1("9. Conclusion"));
paper.push(P("The UAC data support good short-term forecasts of HHS care load. SARIMA and exponential smoothing reach about 95% accuracy and clearly beat persistence at one to four weeks. Machine learning adds the most for discharge demand and early warning, at the price of many false alarms. The main operational risk is a change in regime, not day-to-day noise, so intervals, error monitoring and clear escalation rules matter as much as the point forecast. The dashboard lets planners try different horizons, models, intervals and scenarios using the same tested results."));

paper.push(H1("Appendix: running the code"));
paper.push(P("The project folder contains pipeline.py (data preparation, features, models, walk-forward testing, indicators and final forecasts), app.py (the Streamlit dashboard), make_figures.py, the cleaned data and all test outputs. Install the packages in requirements.txt, run python pipeline.py (about 10 minutes), then run streamlit run app.py. The dashboard lets you choose the forecast horizon, the primary model and which models to overlay, the interval level and basis, the capacity threshold, and a scenario with presets or custom inputs."));

// ================================================================== EXEC SUMMARY
const ex = [];
ex.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Executive Summary", font: FONT, bold: true, size: 40, color: "1F3864" })] }));
ex.push(new Paragraph({ spacing: { after: 160 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "2F5496", space: 4 } }, children: [new TextRun({ text: "Forecasting HHS care load and placement demand, data through 21 December 2025", font: FONT, size: 22, color: "595959" })] }));
ex.push(H2("Where things stand"));
ex.push(P(`About ${f0(D.cur)} children are in HHS care. Our forecasts show the number staying near 2,500 over the next four weeks, with an 80% range of 2,340 to 2,670. Discharges should keep up with incoming transfers. Nothing in the forecasts calls for a scale-up, but care load has climbed about 22% since its August 2025 low, so it makes sense to keep standby arrangements in place.`));
ex.push(H2("What we found"));
ex.push(P("Short-term forecasts are reliable. The best models are about 95% accurate and 12 to 39% better than assuming nothing changes. The typical error is about 115 children at one week and 225 at two weeks, or roughly 2 to 5%."));
ex.push(P("The real danger is a sudden change in regime. Care load fell about 65% between December 2024 and March 2025, and forecast errors during that period were three to five times larger. No model can see a policy-driven shift coming, so planners should keep monitoring and use wide ranges when conditions change."));
ex.push(P("Machine learning is useful for discharges and early warning. It forecast discharge demand 15 to 26% more accurately than the baseline at two to four weeks, and it warned ahead of 15 of 26 rises in care load, about 19 days early on average. About two in three of its warnings were false alarms, so it works best as a signal to watch, not to act on alone."));
ex.push(P("Uncertainty was measured. The 80% intervals were well calibrated in testing. The 95% intervals were slightly too narrow, covering 85 to 90% of outcomes."));
ex.push(H2("Outlook from 21 December 2025"));
ex.push(TBL(["Horizon", "Forecast care load", "80% interval", "95% interval"], [1, 7, 14, 28].map((h) => [h + (h === 1 ? " day" : " days"), f0(pr(h).point), f0(pr(h).lo80) + " to " + f0(pr(h).hi80), f0(pr(h).lo95) + " to " + f0(pr(h).hi95)]), [2000, 2400, 2480, 2480]));
ex.push(CAP("Exponential smoothing, intervals based on the period after the 2025 drop. Discharges for the coming week are forecast at about 80 to 85 children, against about 65 transfers into HHS."));
ex.push(...IMG("fig9_exec", 5.6, "Care-load forecast with 80% and 95% intervals."));
ex.push(H2("What we suggest"));
ex.push(N("Plan on the statistical forecasts: SARIMA for one week and exponential smoothing for two to four weeks."));
ex.push(N("Enter the real capacity threshold in the dashboard and check the breach probability each week."));
ex.push(N("Treat machine-learning surge warnings as a prompt to watch, and escalate if they repeat or if transfers exceed discharges."));
ex.push(N("Track forecast error and switch to wider intervals if it rises well above its recent level of about 2 to 4%."));
ex.push(N("Improve the data by reporting on Fridays and Saturdays and adding other routes into HHS care and policy indicators."));
ex.push(P("A note on limits: the data contain one major break and few surges in the test period, 355 non-reporting days were filled by interpolation, and the scenario figures are planning arithmetic, not causal estimates. More detail is in the research paper.", { size: 19 }));

// ================================================================== build
const styles = { default: { document: { run: { font: FONT, size: 21 } } } };
const numbering = { config: [
  { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
  { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] } ] };
const footer = (t) => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: t + " · Page ", font: FONT, size: 16, color: "7F7F7F" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "7F7F7F" })] })] });
const page = (m) => ({ page: { size: { width: 12240, height: 15840 }, margin: { top: m, right: 1440, bottom: m, left: 1440 } } });
const mk = (children, t, m) => new Document({ styles, numbering, sections: [{ properties: page(m), footers: { default: footer(t) }, children }] });
(async () => {
  fs.mkdirSync("deliverables", { recursive: true });
  fs.writeFileSync("deliverables/UAC_Forecasting_Research_Paper.docx", await Packer.toBuffer(mk(paper, "UAC Care Load & Placement Demand Forecasting", 1300)));
  fs.writeFileSync("deliverables/UAC_Executive_Summary.docx", await Packer.toBuffer(mk(ex, "UAC Forecasting: Executive Summary", 1000)));
  console.log("ok");
})();
