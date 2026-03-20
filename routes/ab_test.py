import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Blueprint, jsonify, render_template, request
from scipy import stats

ab_test_bp = Blueprint("ab_test", __name__)

_FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#374151")
_AXIS = dict(gridcolor="#f0f2f8", linecolor="#e5e7eb", tickfont=dict(size=11, color="#6b7280"), zeroline=False)
_TITLE_FONT = dict(size=14, color="#111827", family="Inter, system-ui, sans-serif")
_LEGEND = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=11))
_MARGIN = dict(t=48, b=44, l=52, r=28)


@ab_test_bp.route("/ab-test")
def ab_test_page():
    return render_template("ab_test.html", active="ab_test")


@ab_test_bp.route("/ab-test/upload-csv", methods=["POST"])
def upload_csv():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No file uploaded"}), 400
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
        required = {"group", "value"}
        if not required.issubset(set(df.columns.str.lower())):
            return jsonify({"error": "CSV must have columns named 'group' and 'value'"}), 400
        df.columns = df.columns.str.lower()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        groups = df["group"].unique().tolist()
        if len(groups) < 2:
            return jsonify({"error": "CSV must contain at least two distinct groups"}), 400
        groups = sorted(groups)[:2]
        g_a = df[df["group"] == groups[0]]["value"]
        g_b = df[df["group"] == groups[1]]["value"]
        return jsonify({
            "n1": len(g_a),
            "mean1": round(float(g_a.mean()), 4),
            "sd1": round(float(g_a.std(ddof=1)), 4),
            "n2": len(g_b),
            "mean2": round(float(g_b.mean()), 4),
            "sd2": round(float(g_b.std(ddof=1)), 4),
            "group_a": str(groups[0]),
            "group_b": str(groups[1]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@ab_test_bp.route("/ab-test/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    try:
        test_type = data.get("test_type", "proportions")
        alpha = float(data.get("alpha", 0.05))
        practical_threshold = float(data.get("practical_threshold", 0.01))

        if test_type == "proportions":
            n1, conv1 = int(data["n1"]), int(data["conv1"])
            n2, conv2 = int(data["n2"]), int(data["conv2"])
            p1, p2 = conv1 / n1, conv2 / n2
            p_pool = (conv1 + conv2) / (n1 + n2)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z_stat = (p2 - p1) / se_pool
            p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
            diff = p2 - p1
            se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            z_crit = stats.norm.ppf(1 - alpha / 2)
            ci_low, ci_high = diff - z_crit * se_diff, diff + z_crit * se_diff
            effect_size = diff / p1 if p1 > 0 else 0
            effect_label = f"Relative lift: {effect_size*100:.1f}%"
            spread = max(abs(p1 - p2) * 5, 0.08)
            x = np.linspace(min(p1, p2) - spread, max(p1, p2) + spread, 500)
            se1 = np.sqrt(p1 * (1 - p1) / n1)
            se2 = np.sqrt(p2 * (1 - p2) / n2)
            y1 = stats.norm.pdf(x, p1, se1)
            y2 = stats.norm.pdf(x, p2, se2)
            stat_label, metric_label = "z-statistic", "Conversion Rate"

        else:
            n1, mean1, sd1 = int(data["n1"]), float(data["mean1"]), float(data["sd1"])
            n2, mean2, sd2 = int(data["n2"]), float(data["mean2"]), float(data["sd2"])
            se = np.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
            t_stat = (mean2 - mean1) / se
            df_num = (sd1 ** 2 / n1 + sd2 ** 2 / n2) ** 2
            df_den = (sd1 ** 2 / n1) ** 2 / (n1 - 1) + (sd2 ** 2 / n2) ** 2 / (n2 - 1)
            df_welch = df_num / df_den
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df_welch)))
            diff = mean2 - mean1
            t_crit = stats.t.ppf(1 - alpha / 2, df_welch)
            ci_low, ci_high = diff - t_crit * se, diff + t_crit * se
            pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
            cohens_d = diff / pooled_sd if pooled_sd > 0 else 0
            effect_size = abs(cohens_d)
            effect_label = f"Cohen's d = {cohens_d:.3f}"
            z_stat = t_stat
            x_range = max(mean1, mean2) + 3.5 * max(sd1, sd2)
            x_min = min(mean1, mean2) - 3.5 * max(sd1, sd2)
            x = np.linspace(x_min, x_range, 500)
            y1 = stats.norm.pdf(x, mean1, sd1)
            y2 = stats.norm.pdf(x, mean2, sd2)
            stat_label, metric_label = "t-statistic", "Outcome"
            p1, p2 = mean1, mean2

        sig = p_value < alpha
        practical_sig = abs(diff) >= practical_threshold

        # CI chart
        fig_ci = go.Figure()
        fig_ci.add_shape(
            type="line", x0=ci_low, x1=ci_high, y0=0, y1=0,
            line=dict(color="#6366f1", width=10),
        )
        fig_ci.add_trace(go.Scatter(
            x=[diff], y=[0],
            mode="markers",
            marker=dict(color="#4f46e5", size=16, symbol="diamond",
                        line=dict(color="#ffffff", width=2.5)),
            name="Observed difference (B − A)",
        ))
        # Zero line
        fig_ci.add_vline(x=0, line_dash="dash", line_color="#f43f5e", line_width=1.5,
                         annotation_text="Zero (no effect)", annotation_position="top",
                         annotation_font=dict(size=11, color="#f43f5e"))
        if abs(practical_threshold) > 0:
            fig_ci.add_vline(x=practical_threshold, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                             annotation_text="Practical threshold", annotation_position="top right",
                             annotation_font=dict(size=11, color="#f59e0b"))
        fig_ci.update_layout(
            template=None, paper_bgcolor="#ffffff", plot_bgcolor="#fafbff",
            font=_FONT, margin=dict(t=44, b=44, l=28, r=28),
            height=220,
            title=dict(text=f"{int((1-alpha)*100)}% Confidence Interval for Difference (B − A)",
                       font=_TITLE_FONT, x=0, xanchor="left"),
            xaxis=dict(title=dict(text="Difference", font=dict(size=12, color="#6b7280")), **_AXIS),
            yaxis=dict(visible=False),
            showlegend=True,
            legend=dict(orientation="h", y=-0.25, **_LEGEND),
        )

        # Distribution overlap
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=x.tolist(), y=y1.tolist(), mode="lines",
            name="Group A — Control",
            line=dict(color="#6366f1", width=2),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.14)",
        ))
        fig_dist.add_trace(go.Scatter(
            x=x.tolist(), y=y2.tolist(), mode="lines",
            name="Group B — Treatment",
            line=dict(color="#f43f5e", width=2),
            fill="tozeroy", fillcolor="rgba(244,63,94,0.12)",
        ))
        fig_dist.update_layout(
            template=None, paper_bgcolor="#ffffff", plot_bgcolor="#fafbff",
            font=_FONT, margin=_MARGIN, height=340,
            title=dict(text="Sampling Distributions of Group Estimates",
                       font=_TITLE_FONT, x=0, xanchor="left"),
            xaxis=dict(title=dict(text=metric_label, font=dict(size=12, color="#6b7280")), **_AXIS),
            yaxis=dict(title=dict(text="Density", font=dict(size=12, color="#6b7280")), **_AXIS),
            legend=dict(x=0.02, y=0.98, **_LEGEND),
        )

        stat_sig_text = "statistically significant" if sig else "NOT statistically significant"
        prac_sig_text = "practically significant" if practical_sig else "NOT practically significant"

        if test_type == "proportions":
            interp = (
                f"Group B converted at <strong>{p2:.2%}</strong> vs Group A at "
                f"<strong>{p1:.2%}</strong> (difference: {diff:+.4f}). "
                f"The {stat_label} is <strong>{z_stat:.3f}</strong> (p = <strong>{p_value:.4f}</strong>). "
                f"The result is <strong>{stat_sig_text}</strong> at α = {alpha}. "
                f"Effect size — {effect_label}. "
                f"The {int((1-alpha)*100)}% CI for B − A is [{ci_low:.4f}, {ci_high:.4f}]. "
                f"Based on your practical threshold of {practical_threshold:.3f}, "
                f"the effect is <strong>{prac_sig_text}</strong>."
            )
        else:
            interp = (
                f"Group B has mean <strong>{mean2:.3f}</strong> vs Group A mean "
                f"<strong>{mean1:.3f}</strong> (difference: {diff:+.3f}). "
                f"The {stat_label} is <strong>{z_stat:.3f}</strong> (p = <strong>{p_value:.4f}</strong>). "
                f"The result is <strong>{stat_sig_text}</strong> at α = {alpha}. "
                f"Effect size — {effect_label}. "
                f"The {int((1-alpha)*100)}% CI for B − A is [{ci_low:.3f}, {ci_high:.3f}]. "
                f"Based on your practical threshold of {practical_threshold}, "
                f"the effect is <strong>{prac_sig_text}</strong>."
            )

        if sig and not practical_sig:
            interp += (
                " <strong>Note:</strong> The result is statistically significant but does not meet your "
                "practical threshold — the effect is real but may not be worth acting on."
            )
        elif not sig and practical_sig:
            interp += (
                " <strong>Note:</strong> The observed effect exceeds your practical threshold, "
                "but is not statistically significant — collect more data before concluding."
            )

        return jsonify({
            "z_stat": round(float(z_stat), 4),
            "p_value": round(float(p_value), 6),
            "stat_sig": bool(sig),
            "practical_sig": bool(practical_sig),
            "effect_label": effect_label,
            "ci_low": round(float(ci_low), 6),
            "ci_high": round(float(ci_high), 6),
            "stat_label": stat_label,
            "chart_ci": json.dumps(fig_ci, cls=plotly.utils.PlotlyJSONEncoder),
            "chart_dist": json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder),
            "interpretation": interp,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
