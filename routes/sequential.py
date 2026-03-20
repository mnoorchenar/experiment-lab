import json

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Blueprint, jsonify, render_template, request
from scipy import stats

sequential_bp = Blueprint("sequential", __name__)

_FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#374151")
_AXIS = dict(gridcolor="#f0f2f8", linecolor="#e5e7eb", tickfont=dict(size=11, color="#6b7280"), zeroline=False)
_TITLE_FONT = dict(size=14, color="#111827", family="Inter, system-ui, sans-serif")
_LEGEND = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=11))
_MARGIN = dict(t=52, b=48, l=60, r=100)


@sequential_bp.route("/sequential")
def sequential_page():
    return render_template("sequential.html", active="sequential")


@sequential_bp.route("/sequential/simulate", methods=["POST"])
def simulate():
    data = request.get_json()
    try:
        true_effect = float(data.get("true_effect", 0.0))
        obs_per_step = int(data.get("obs_per_step", 50))
        total_steps = int(data.get("total_steps", 20))
        alpha = float(data.get("alpha", 0.05))
        num_sims = min(int(data.get("num_sims", 30)), 50)

        np.random.seed(42)
        z_final = stats.norm.ppf(1 - alpha / 2)
        steps = list(range(1, total_steps + 1))

        # O'Brien-Fleming: boundary at information fraction t is z_final / sqrt(t)
        obf_boundaries = [float(2 * (1 - stats.norm.cdf(z_final / np.sqrt(s / total_steps)))) for s in steps]

        all_pvalues, naive_crosses, seq_crosses = [], [], []

        for _ in range(num_sims):
            obs_a_all, obs_b_all, run_pvals = [], [], []
            for step in steps:
                obs_a_all.extend(np.random.normal(0, 1, obs_per_step).tolist())
                obs_b_all.extend(np.random.normal(true_effect, 1, obs_per_step).tolist())
                n = len(obs_a_all)
                mean_diff = np.mean(obs_b_all) - np.mean(obs_a_all)
                z = mean_diff / np.sqrt(2.0 / n)
                p = float(2 * (1 - stats.norm.cdf(abs(z))))
                run_pvals.append(p)
            all_pvalues.append(run_pvals)
            naive_crosses.append(any(p <= alpha for p in run_pvals))
            seq_crosses.append(any(p <= b for p, b in zip(run_pvals, obf_boundaries)))

        naive_fp_rate = sum(naive_crosses) / num_sims
        seq_fp_rate = sum(seq_crosses) / num_sims

        # --- Chart 1: Naive repeated testing ---
        fig_naive = go.Figure()
        for i, run_pvals in enumerate(all_pvalues):
            color = "rgba(244,63,94,0.35)" if naive_crosses[i] else "rgba(99,102,241,0.22)"
            fig_naive.add_trace(go.Scatter(
                x=steps, y=run_pvals, mode="lines",
                line=dict(color=color, width=1.2),
                showlegend=False, hoverinfo="none",
            ))
        # Legend proxies
        fig_naive.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                       line=dict(color="rgba(244,63,94,0.75)", width=2.5),
                                       name="Crossed α (false positive)"))
        fig_naive.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                       line=dict(color="rgba(99,102,241,0.6)", width=2.5),
                                       name="Did not cross α"))
        fig_naive.add_hline(y=alpha, line_dash="dash", line_color="#f43f5e", line_width=2,
                            annotation_text=f"α = {alpha}",
                            annotation_position="right",
                            annotation_font=dict(size=11, color="#f43f5e"))
        fig_naive.update_layout(
            template=None, paper_bgcolor="#ffffff", plot_bgcolor="#fafbff",
            font=_FONT, margin=_MARGIN, height=400,
            title=dict(text=f"Naive Repeated Testing — Running p-values ({num_sims} simulations)",
                       font=_TITLE_FONT, x=0, xanchor="left"),
            xaxis=dict(title=dict(text="Number of Interim Looks", font=dict(size=12, color="#6b7280")), **_AXIS),
            yaxis=dict(title=dict(text="p-value", font=dict(size=12, color="#6b7280")), range=[0, 1], **_AXIS),
            legend=dict(x=0.01, y=0.98, **_LEGEND),
        )

        # --- Chart 2: Sequential with OBF boundary ---
        fig_seq = go.Figure()
        for i, run_pvals in enumerate(all_pvalues):
            color = "rgba(244,63,94,0.35)" if seq_crosses[i] else "rgba(99,102,241,0.22)"
            fig_seq.add_trace(go.Scatter(
                x=steps, y=run_pvals, mode="lines",
                line=dict(color=color, width=1.2),
                showlegend=False, hoverinfo="none",
            ))
        # OBF boundary — make it the hero line
        fig_seq.add_trace(go.Scatter(
            x=steps, y=obf_boundaries, mode="lines",
            name="O'Brien-Fleming boundary",
            line=dict(color="#10b981", width=3, dash="dash"),
        ))
        fig_seq.add_hline(y=alpha, line_dash="dot", line_color="#f43f5e", line_width=1.5,
                          annotation_text=f"Naive α = {alpha}",
                          annotation_position="right",
                          annotation_font=dict(size=11, color="#f43f5e"))
        fig_seq.update_layout(
            template=None, paper_bgcolor="#ffffff", plot_bgcolor="#fafbff",
            font=_FONT, margin=_MARGIN, height=400,
            title=dict(text=f"Sequential Testing with O'Brien-Fleming Boundary ({num_sims} simulations)",
                       font=_TITLE_FONT, x=0, xanchor="left"),
            xaxis=dict(title=dict(text="Number of Interim Looks", font=dict(size=12, color="#6b7280")), **_AXIS),
            yaxis=dict(title=dict(text="p-value", font=dict(size=12, color="#6b7280")), range=[0, 1], **_AXIS),
            legend=dict(x=0.01, y=0.98, **_LEGEND),
        )

        # --- Chart 3: Comparison bar ---
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=["Naive Repeated Testing", "Sequential (O'Brien-Fleming)"],
            y=[naive_fp_rate * 100, seq_fp_rate * 100],
            marker=dict(
                color=["#f43f5e", "#10b981"],
                line=dict(color=["#e11d48", "#059669"], width=1.5),
            ),
            text=[f"{naive_fp_rate*100:.1f}%", f"{seq_fp_rate*100:.1f}%"],
            textposition="outside",
            textfont=dict(size=14, color="#374151"),
            width=0.45,
        ))
        fig_compare.add_hline(y=alpha * 100, line_dash="dash", line_color="#6b7280", line_width=1.5,
                              annotation_text=f"Nominal α = {alpha*100:.0f}%",
                              annotation_font=dict(size=11, color="#6b7280"))
        fig_compare.update_layout(
            template=None, paper_bgcolor="#ffffff", plot_bgcolor="#fafbff",
            font=_FONT, margin=dict(t=48, b=48, l=56, r=24), height=350,
            title=dict(text="False Positive Rate: Naive vs Sequential Testing",
                       font=_TITLE_FONT, x=0, xanchor="left"),
            yaxis=dict(title=dict(text="False Positive Rate (%)", font=dict(size=12, color="#6b7280")),
                       range=[0, max(naive_fp_rate * 100 + 15, 40)], **_AXIS),
            xaxis=dict(**_AXIS),
            bargap=0.5, showlegend=False,
        )

        effect_desc = (
            "no true effect (true difference = 0)"
            if true_effect == 0
            else f"a true effect of {true_effect:.2f} standard deviations"
        )
        interpretation = (
            f"Across <strong>{num_sims} simulated experiments</strong> with {effect_desc}, "
            f"each peeked at <strong>{total_steps} interim points</strong> ({obs_per_step} new observations per look):"
            f"<br><br>"
            f"<strong>Naive repeated testing</strong> crossed α = {alpha} at least once in "
            f"<strong>{naive_fp_rate*100:.1f}%</strong> of simulations. "
        )
        if true_effect == 0:
            interpretation += (
                f"Since there is no real effect, every one of these crossings is a false positive — ideally "
                f"this rate should be {alpha*100:.0f}%. Peeking inflates it dramatically."
            )
        else:
            interpretation += f"With a true effect present, this reflects the detection rate (power) of the naive approach."

        interpretation += (
            f"<br><br>"
            f"<strong>Sequential testing (O'Brien-Fleming)</strong> controlled the false positive rate at "
            f"<strong>{seq_fp_rate*100:.1f}%</strong> — much closer to the nominal {alpha*100:.0f}%. "
            f"The OBF boundary demands a more extreme result early in the experiment (very low p-value) "
            f"and relaxes as more data accumulates, enabling valid early stopping. "
            f"The trade-off: you need a slightly more extreme signal to stop early — "
            f"a small price for statistical validity."
        )

        return jsonify({
            "naive_fp_rate": round(naive_fp_rate * 100, 1),
            "seq_fp_rate": round(seq_fp_rate * 100, 1),
            "chart_naive": json.dumps(fig_naive, cls=plotly.utils.PlotlyJSONEncoder),
            "chart_seq": json.dumps(fig_seq, cls=plotly.utils.PlotlyJSONEncoder),
            "chart_compare": json.dumps(fig_compare, cls=plotly.utils.PlotlyJSONEncoder),
            "interpretation": interpretation,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400
