import json

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Blueprint, jsonify, render_template, request
from scipy import stats

sequential_bp = Blueprint("sequential", __name__)


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
        num_sims = min(int(data.get("num_sims", 25)), 50)

        np.random.seed(42)
        z_final = stats.norm.ppf(1 - alpha / 2)
        steps = list(range(1, total_steps + 1))

        # O'Brien-Fleming boundaries (p-value scale)
        # At information fraction t, OBF z-boundary ≈ z_final / sqrt(t)
        obf_boundaries = []
        for step in steps:
            t = step / total_steps
            z_b = z_final / np.sqrt(t)
            obf_boundaries.append(float(2 * (1 - stats.norm.cdf(z_b))))

        # Simulate experiments
        all_pvalues = []
        naive_crosses = []
        seq_crosses = []

        for _ in range(num_sims):
            obs_a_all = []
            obs_b_all = []
            run_pvals = []

            for step in steps:
                obs_a_all.extend(np.random.normal(0, 1, obs_per_step).tolist())
                obs_b_all.extend(np.random.normal(true_effect, 1, obs_per_step).tolist())

                n = len(obs_a_all)
                mean_diff = np.mean(obs_b_all) - np.mean(obs_a_all)
                se = np.sqrt(2.0 / n)
                z = mean_diff / se
                p = float(2 * (1 - stats.norm.cdf(abs(z))))
                run_pvals.append(p)

            all_pvalues.append(run_pvals)
            naive_crosses.append(any(p <= alpha for p in run_pvals))
            seq_crosses.append(
                any(p <= b for p, b in zip(run_pvals, obf_boundaries))
            )

        naive_fp_rate = sum(naive_crosses) / num_sims
        seq_fp_rate = sum(seq_crosses) / num_sims

        # Chart 1 — Naive repeated testing
        fig_naive = go.Figure()
        for i, run_pvals in enumerate(all_pvalues):
            color = "rgba(239,68,68,0.35)" if naive_crosses[i] else "rgba(99,102,241,0.25)"
            fig_naive.add_trace(
                go.Scatter(
                    x=steps,
                    y=run_pvals,
                    mode="lines",
                    line=dict(color=color, width=1.2),
                    showlegend=False,
                    hoverinfo="none",
                )
            )
        # Add legend proxies
        fig_naive.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                       line=dict(color="rgba(239,68,68,0.7)", width=2),
                                       name="Crossed α (false positive)"))
        fig_naive.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                       line=dict(color="rgba(99,102,241,0.6)", width=2),
                                       name="Did not cross α"))
        fig_naive.add_hline(
            y=alpha,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text=f"α = {alpha}",
            annotation_position="right",
        )
        fig_naive.update_layout(
            title=f"Naive Repeated Testing — Running p-values ({num_sims} simulations)",
            xaxis_title="Number of Interim Looks",
            yaxis_title="p-value",
            template="plotly_white",
            height=400,
            margin=dict(t=60, b=50, l=60, r=120),
            yaxis=dict(range=[0, 1]),
            legend=dict(x=0.55, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        )

        # Chart 2 — Sequential testing with OBF boundary
        fig_seq = go.Figure()
        for i, run_pvals in enumerate(all_pvalues):
            color = "rgba(239,68,68,0.35)" if seq_crosses[i] else "rgba(99,102,241,0.25)"
            fig_seq.add_trace(
                go.Scatter(
                    x=steps,
                    y=run_pvals,
                    mode="lines",
                    line=dict(color=color, width=1.2),
                    showlegend=False,
                    hoverinfo="none",
                )
            )
        fig_seq.add_trace(
            go.Scatter(
                x=steps,
                y=obf_boundaries,
                mode="lines",
                name="O'Brien-Fleming boundary",
                line=dict(color="#22c55e", width=2.5, dash="dash"),
            )
        )
        fig_seq.add_hline(
            y=alpha,
            line_dash="dot",
            line_color="#ef4444",
            line_width=1.5,
            annotation_text=f"Naive α = {alpha}",
            annotation_position="right",
        )
        fig_seq.update_layout(
            title=f"Sequential Testing with O'Brien-Fleming Boundary ({num_sims} simulations)",
            xaxis_title="Number of Interim Looks",
            yaxis_title="p-value",
            template="plotly_white",
            height=400,
            margin=dict(t=60, b=50, l=60, r=120),
            yaxis=dict(range=[0, 1]),
            legend=dict(x=0.5, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        )

        # Chart 3 — Comparison bar chart
        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Bar(
                x=["Naive Repeated Testing", "Sequential (O'Brien-Fleming)"],
                y=[naive_fp_rate * 100, seq_fp_rate * 100],
                marker_color=["#ef4444", "#22c55e"],
                text=[f"{naive_fp_rate*100:.1f}%", f"{seq_fp_rate*100:.1f}%"],
                textposition="outside",
                width=0.5,
            )
        )
        fig_compare.add_hline(
            y=alpha * 100,
            line_dash="dash",
            line_color="#6b7280",
            annotation_text=f"Nominal α = {alpha*100:.0f}%",
        )
        fig_compare.update_layout(
            title="False Positive Rate: Naive vs Sequential Testing",
            yaxis_title="False Positive Rate (%)",
            template="plotly_white",
            height=350,
            margin=dict(t=50, b=50, l=60, r=30),
            yaxis=dict(range=[0, max(naive_fp_rate * 100 + 15, 40)]),
            showlegend=False,
        )

        effect_desc = (
            "no true effect (true difference = 0)"
            if true_effect == 0
            else f"a true effect of {true_effect:.2f} standard deviations"
        )
        interpretation = (
            f"Across <strong>{num_sims} simulated experiments</strong> with {effect_desc}, "
            f"each peeked at {total_steps} interim points ({obs_per_step} new observations per look):"
            f"<br><br>"
            f"<strong>Naive repeated testing</strong> crossed α = {alpha} at least once in "
            f"<strong>{naive_fp_rate*100:.1f}%</strong> of simulations. "
        )
        if true_effect == 0:
            interpretation += (
                f"Since there is no real effect, every one of these crossings is a false positive. "
                f"Ideally this rate should be {alpha*100:.0f}% — peeking inflates it dramatically."
            )
        else:
            interpretation += (
                f"With a true effect present, this measures the detection rate (power)."
            )
        interpretation += (
            f"<br><br>"
            f"<strong>Sequential testing with the O'Brien-Fleming boundary</strong> "
            f"controlled the false positive rate at <strong>{seq_fp_rate*100:.1f}%</strong> — "
            f"much closer to the nominal {alpha*100:.0f}%. "
            f"The OBF boundary demands a much more extreme result early in the experiment "
            f"(very low p-value) and relaxes gradually as more data accumulates, "
            f"allowing valid early stopping without inflating the type I error rate. "
            f"The trade-off: you need a slightly more extreme signal to stop early."
        )

        return jsonify(
            {
                "naive_fp_rate": round(naive_fp_rate * 100, 1),
                "seq_fp_rate": round(seq_fp_rate * 100, 1),
                "chart_naive": json.dumps(fig_naive, cls=plotly.utils.PlotlyJSONEncoder),
                "chart_seq": json.dumps(fig_seq, cls=plotly.utils.PlotlyJSONEncoder),
                "chart_compare": json.dumps(fig_compare, cls=plotly.utils.PlotlyJSONEncoder),
                "interpretation": interpretation,
            }
        )
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400
