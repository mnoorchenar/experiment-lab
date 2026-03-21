import json
import math

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Blueprint, jsonify, render_template, request
from scipy import stats

power_bp = Blueprint("power", __name__)

# Shared Plotly style constants
_FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#374151")
_AXIS = dict(gridcolor="#f0f2f8", linecolor="#e5e7eb", tickfont=dict(size=11, color="#6b7280"), zeroline=False)
_LEGEND = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=11))
_MARGIN = dict(t=16, b=40, l=56, r=20)
_HOVERLABEL = dict(
    bgcolor="#1e1b4b",
    bordercolor="#4f46e5",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#e0e7ff"),
)


def _base_layout(**kwargs):
    return dict(
        template=None,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafbff",
        font=_FONT,
        margin=_MARGIN,
        hovermode="x unified",
        hoverlabel=_HOVERLABEL,
        **kwargs,
    )


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

        # Power curve
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
        fig1.add_trace(go.Scatter(
            x=effect_sizes.tolist(), y=powers_curve,
            mode="lines", name="Statistical Power",
            line=dict(color="#6366f1", width=2.5),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        ))
        fig1.add_hline(
            y=power_target, line_dash="dash", line_color="#10b981", line_width=1.75,
            annotation_text=f"Target Power = {power_target:.0%}",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#10b981"),
        )
        fig1.add_vline(
            x=mde, line_dash="dash", line_color="#f43f5e", line_width=1.75,
            annotation_text=f"Your MDE = {mde:.3f}",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#f43f5e"),
        )
        fig1.update_layout(
            **_base_layout(height=360),
            xaxis=dict(title=dict(text="Effect Size (absolute difference in proportions)", font=dict(size=12, color="#6b7280")), **_AXIS),
            yaxis=dict(title=dict(text="Statistical Power", font=dict(size=12, color="#6b7280")), range=[0, 1.05], tickformat=".0%", **_AXIS),
            legend=dict(x=0.02, y=0.98, **_LEGEND),
        )

        # Sample size vs power curve
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
        fig2.add_trace(go.Scatter(
            x=power_range.tolist(), y=sample_sizes_curve,
            mode="lines", name="Sample Size per Group",
            line=dict(color="#4f46e5", width=2.5),
            fill="tozeroy", fillcolor="rgba(79,70,229,0.09)",
        ))
        fig2.add_vline(
            x=power_target, line_dash="dash", line_color="#f43f5e", line_width=1.75,
            annotation_text=f"Your Power = {power_target:.0%}",
            annotation_font=dict(size=11, color="#f43f5e"),
        )
        fig2.update_layout(
            **_base_layout(height=360),
            xaxis=dict(title=dict(text="Desired Power (1 − β)", font=dict(size=12, color="#6b7280")), tickformat=".0%", **_AXIS),
            yaxis=dict(title=dict(text="Required Sample Size (per group)", font=dict(size=12, color="#6b7280")), **_AXIS),
        )

        pct_change = mde / baseline * 100 if baseline > 0 else 0
        interpretation = (
            f"To detect a lift of <strong>{mde*100:.1f}%</strong> from a baseline of "
            f"<strong>{baseline*100:.1f}%</strong> ({pct_change:.1f}% relative lift), "
            f"you need <strong>{n:,} observations per group</strong> "
            f"({n * 2:,} total) to achieve <strong>{power_target*100:.0f}% power</strong> "
            f"at α = {alpha} ({tails}-tailed). "
            f"If the true effect is at least {mde*100:.1f}%, you will detect it "
            f"{power_target*100:.0f}% of the time. You will falsely claim significance "
            f"only {alpha*100:.0f}% of the time when there is no real effect."
        )

        return jsonify({
            "sample_size": n,
            "chart1": json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
            "chart2": json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
            "interpretation": interpretation,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
