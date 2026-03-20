import json
import math

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Blueprint, jsonify, render_template, request
from scipy import stats

power_bp = Blueprint("power", __name__)


@power_bp.route("/power")
def power_page():
    return render_template("power.html", active="power")


@power_bp.route("/power/calculate", methods=["POST"])
def calculate_power():
    data = request.get_json()
    try:
        baseline = float(data.get("baseline", 0.10))
        mde = float(data.get("mde", 0.02))
        alpha = float(data.get("alpha", 0.05))
        power_target = float(data.get("power", 0.80))
        tails = int(data.get("tails", 2))

        p1 = min(max(baseline, 0.001), 0.999)
        p2 = min(max(baseline + mde, 0.001), 0.999)

        if tails == 2:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power_target)
        p_bar = (p1 + p2) / 2

        numerator = (
            z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        denominator = (p2 - p1) ** 2
        n = math.ceil(numerator / denominator)

        # Power curve: vary effect size at fixed n
        max_es = min(mde * 4, (1 - baseline) * 0.8)
        effect_sizes = np.linspace(mde * 0.05, max_es, 300)
        powers_curve = []
        for es in effect_sizes:
            p2_t = min(p1 + es, 0.999)
            pb = (p1 + p2_t) / 2
            ncp = es / math.sqrt(pb * (1 - pb) * 2 / n) if n > 0 else 0
            if tails == 2:
                pw = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
            else:
                pw = 1 - stats.norm.cdf(z_alpha - ncp)
            powers_curve.append(max(0.0, min(1.0, float(pw))))

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=effect_sizes.tolist(),
                y=powers_curve,
                mode="lines",
                name="Statistical Power",
                line=dict(color="#4f46e5", width=2.5),
            )
        )
        fig1.add_hline(
            y=power_target,
            line_dash="dash",
            line_color="#22c55e",
            annotation_text=f"Target Power = {power_target:.0%}",
            annotation_position="bottom right",
        )
        fig1.add_vline(
            x=mde,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Your MDE = {mde:.3f}",
            annotation_position="top left",
        )
        fig1.update_layout(
            title="Statistical Power vs Effect Size",
            xaxis_title="Effect Size (absolute difference in proportions)",
            yaxis_title="Statistical Power",
            template="plotly_white",
            height=380,
            margin=dict(t=50, b=50, l=60, r=20),
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            legend=dict(x=0.7, y=0.1),
        )

        # Sample size vs desired power curve
        power_range = np.linspace(0.50, 0.99, 150)
        sample_sizes_curve = []
        for pw in power_range:
            zb = stats.norm.ppf(pw)
            num = (
                z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
            ) ** 2
            sample_sizes_curve.append(math.ceil(num / denominator))

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=power_range.tolist(),
                y=sample_sizes_curve,
                mode="lines",
                name="Sample Size per Group",
                line=dict(color="#3b82f6", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.08)",
            )
        )
        fig2.add_vline(
            x=power_target,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Your Power = {power_target:.0%}",
        )
        fig2.update_layout(
            title="Required Sample Size vs Desired Power",
            xaxis_title="Desired Power (1 - β)",
            yaxis_title="Required Sample Size (per group)",
            template="plotly_white",
            height=380,
            margin=dict(t=50, b=50, l=70, r=20),
            xaxis=dict(tickformat=".0%"),
        )

        pct_change = mde / baseline * 100 if baseline > 0 else 0
        interpretation = (
            f"To detect a lift of <strong>{mde*100:.1f}%</strong> from a baseline of "
            f"<strong>{baseline*100:.1f}%</strong> ({pct_change:.1f}% relative lift), "
            f"you need <strong>{n:,} observations per group</strong> "
            f"({n * 2:,} total across both groups) with <strong>{power_target*100:.0f}% power</strong> "
            f"at a significance level of α = {alpha} ({tails}-tailed). "
            f"If the true effect is at least {mde*100:.1f}%, you will detect it "
            f"{power_target*100:.0f}% of the time. You will falsely claim significance "
            f"only {alpha*100:.0f}% of the time when there is no real effect."
        )

        return jsonify(
            {
                "sample_size": n,
                "chart1": json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
                "chart2": json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
                "interpretation": interpretation,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400
