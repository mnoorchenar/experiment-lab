import json

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Blueprint, jsonify, render_template, request
from scipy import stats

doe_bp = Blueprint("doe", __name__)

_FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#374151")
_AXIS = dict(gridcolor="#f0f2f8", linecolor="#e5e7eb", tickfont=dict(size=11, color="#6b7280"), zeroline=False)
_LEGEND = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=11))
_MARGIN = dict(t=16, b=40, l=64, r=20)
_HOVERLABEL = dict(
    bgcolor="#1e1b4b",
    bordercolor="#4f46e5",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#e0e7ff"),
)

# Rich, distinct palette
_COLORS = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]


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


@doe_bp.route("/doe")
def doe_page():
    return render_template("doe.html", active="doe")


@doe_bp.route("/doe/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    try:
        factor_a = data.get("factor_a", "Factor A")
        factor_b = data.get("factor_b", "Factor B")
        levels_a = int(data.get("levels_a", 2))
        levels_b = int(data.get("levels_b", 2))
        response_name = data.get("response_name", "Response")
        response_grid = data.get("response_grid", [])

        observations = []
        for i in range(levels_a):
            for j in range(levels_b):
                cell = response_grid[i][j]
                if isinstance(cell, list):
                    for v in cell:
                        observations.append((i, j, float(v)))
                else:
                    observations.append((i, j, float(cell)))

        if len(observations) < levels_a * levels_b:
            return jsonify({"error": "Incomplete response grid"}), 400

        a_idx = np.array([o[0] for o in observations])
        b_idx = np.array([o[1] for o in observations])
        y = np.array([o[2] for o in observations])
        n_total = len(y)
        grand_mean = float(np.mean(y))

        cell_means = np.zeros((levels_a, levels_b))
        cell_counts = np.zeros((levels_a, levels_b), dtype=int)
        for i in range(levels_a):
            for j in range(levels_b):
                mask = (a_idx == i) & (b_idx == j)
                if mask.sum() > 0:
                    cell_means[i, j] = float(np.mean(y[mask]))
                    cell_counts[i, j] = int(mask.sum())

        a_means = cell_means.mean(axis=1)
        b_means = cell_means.mean(axis=0)
        ss_total = float(np.sum((y - grand_mean) ** 2))

        ss_error = 0.0
        for i in range(levels_a):
            for j in range(levels_b):
                mask = (a_idx == i) & (b_idx == j)
                if mask.sum() > 1:
                    ss_error += float(np.sum((y[mask] - cell_means[i, j]) ** 2))

        ss_a = float(sum(int(cell_counts[i, :].sum()) * (a_means[i] - grand_mean) ** 2 for i in range(levels_a)))
        ss_b = float(sum(int(cell_counts[:, j].sum()) * (b_means[j] - grand_mean) ** 2 for j in range(levels_b)))
        ss_cells = float(sum(
            cell_counts[i, j] * (cell_means[i, j] - grand_mean) ** 2
            for i in range(levels_a) for j in range(levels_b)
        ))
        ss_ab = ss_cells - ss_a - ss_b

        df_a, df_b = levels_a - 1, levels_b - 1
        df_ab = df_a * df_b
        df_error = n_total - levels_a * levels_b
        df_total = n_total - 1
        ms_a, ms_b = ss_a / df_a, ss_b / df_b
        ms_ab = ss_ab / df_ab if df_ab > 0 else 0.0

        replicated = df_error > 0
        if replicated:
            ms_error = ss_error / df_error
            f_a = ms_a / ms_error if ms_error > 0 else float("nan")
            f_b = ms_b / ms_error if ms_error > 0 else float("nan")
            f_ab = ms_ab / ms_error if ms_error > 0 else float("nan")
            p_a = float(1 - stats.f.cdf(f_a, df_a, df_error)) if not np.isnan(f_a) else float("nan")
            p_b = float(1 - stats.f.cdf(f_b, df_b, df_error)) if not np.isnan(f_b) else float("nan")
            p_ab = float(1 - stats.f.cdf(f_ab, df_ab, df_error)) if not np.isnan(f_ab) else float("nan")
        else:
            ms_error = None
            f_a = ms_a / ms_ab if ms_ab > 0 and df_ab > 0 else float("nan")
            f_b = ms_b / ms_ab if ms_ab > 0 and df_ab > 0 else float("nan")
            f_ab = float("nan")
            p_a = float(1 - stats.f.cdf(f_a, df_a, df_ab)) if not np.isnan(f_a) else float("nan")
            p_b = float(1 - stats.f.cdf(f_b, df_b, df_ab)) if not np.isnan(f_b) else float("nan")
            p_ab = float("nan")

        def fmt_p(v):
            if np.isnan(v): return "N/A"
            if v < 0.001: return "< 0.001"
            return f"{v:.4f}"

        def fmt_f(v):
            return "N/A" if np.isnan(v) else f"{v:.3f}"

        def fmt_ss(v):
            return f"{v:.4f}"

        anova_table = [
            {"source": factor_a, "ss": fmt_ss(ss_a), "df": df_a, "ms": fmt_ss(ms_a), "f": fmt_f(f_a), "p": fmt_p(p_a), "sig": (not np.isnan(p_a)) and p_a < 0.05},
            {"source": factor_b, "ss": fmt_ss(ss_b), "df": df_b, "ms": fmt_ss(ms_b), "f": fmt_f(f_b), "p": fmt_p(p_b), "sig": (not np.isnan(p_b)) and p_b < 0.05},
            {"source": f"{factor_a} \u00d7 {factor_b}", "ss": fmt_ss(ss_ab), "df": df_ab, "ms": fmt_ss(ms_ab), "f": fmt_f(f_ab), "p": fmt_p(p_ab), "sig": (not np.isnan(p_ab)) and p_ab < 0.05},
        ]
        if replicated:
            anova_table.append({"source": "Error (Within)", "ss": fmt_ss(ss_error), "df": df_error, "ms": fmt_ss(ms_error), "f": "", "p": "", "sig": False})
        anova_table.append({"source": "Total", "ss": fmt_ss(ss_total), "df": df_total, "ms": "", "f": "", "p": "", "sig": False})

        a_labels = [f"{factor_a} \u2014 L{i+1}" for i in range(levels_a)]
        b_labels = [f"{factor_b} \u2014 L{j+1}" for j in range(levels_b)]

        # Interaction plot
        fig_interaction = go.Figure()
        for j in range(levels_b):
            fig_interaction.add_trace(go.Scatter(
                x=a_labels, y=cell_means[:, j].tolist(),
                mode="lines+markers",
                name=b_labels[j],
                line=dict(color=_COLORS[j % len(_COLORS)], width=2.5),
                marker=dict(size=11, line=dict(color="#ffffff", width=2)),
            ))
        fig_interaction.update_layout(
            **_base_layout(height=400),
            xaxis=dict(title=dict(text=factor_a, font=dict(size=12, color="#6b7280")), **_AXIS),
            yaxis=dict(title=dict(text=f"Mean {response_name}", font=dict(size=12, color="#6b7280")), **_AXIS),
            legend=dict(title=dict(text=factor_b, font=dict(size=12, color="#374151")), x=0.02, y=0.98, **_LEGEND),
        )

        # Main effects bar chart
        fig_main = go.Figure()
        fig_main.add_trace(go.Bar(
            x=a_labels, y=a_means.tolist(),
            name=factor_a,
            marker=dict(color=_COLORS[0], line=dict(color="#4338ca", width=1.5)),
            text=[f"{v:.3f}" for v in a_means],
            textposition="outside",
            textfont=dict(size=12, color="#374151"),
            width=0.45,
        ))
        fig_main.add_hline(
            y=grand_mean, line_dash="dash", line_color="#6b7280", line_width=1.5,
            annotation_text=f"Grand Mean = {grand_mean:.3f}",
            annotation_font=dict(size=11, color="#6b7280"),
        )
        fig_main.update_layout(
            **_base_layout(height=350),
            xaxis=dict(title=dict(text=factor_a, font=dict(size=12, color="#6b7280")), **_AXIS),
            yaxis=dict(title=dict(text=f"Mean {response_name}", font=dict(size=12, color="#6b7280")), **_AXIS),
            bargap=0.4, showlegend=False,
        )

        sig_a = (not np.isnan(p_a)) and p_a < 0.05
        sig_b = (not np.isnan(p_b)) and p_b < 0.05
        sig_ab = (not np.isnan(p_ab)) and p_ab < 0.05

        parts = []
        if sig_a:
            best_a_i = int(np.argmax(a_means))
            parts.append(f"<strong>{factor_a}</strong> has a significant main effect (p = {fmt_p(p_a)}). The highest mean {response_name} is at <strong>{a_labels[best_a_i]}</strong> ({a_means[best_a_i]:.3f}).")
        else:
            parts.append(f"<strong>{factor_a}</strong> does not have a significant main effect (p = {fmt_p(p_a)}).")

        if sig_b:
            best_b_j = int(np.argmax(b_means))
            parts.append(f"<strong>{factor_b}</strong> has a significant main effect (p = {fmt_p(p_b)}). The highest mean {response_name} is at <strong>{b_labels[best_b_j]}</strong> ({b_means[best_b_j]:.3f}).")
        else:
            parts.append(f"<strong>{factor_b}</strong> does not have a significant main effect (p = {fmt_p(p_b)}).")

        if sig_ab:
            parts.append(f"The <strong>interaction between {factor_a} and {factor_b}</strong> is significant (p = {fmt_p(p_ab)}). This means the effect of {factor_a} on {response_name} depends on which level of {factor_b} is used \u2014 interpret the interaction plot, not the main effects in isolation. Non-parallel lines in the plot confirm this interaction visually.")
        else:
            parts.append(f"The <strong>interaction between {factor_a} and {factor_b}</strong> is not significant (p = {fmt_p(p_ab)}), meaning the factors act independently on {response_name}. Main effects can be interpreted directly.")

        best_idx = np.unravel_index(np.argmax(cell_means), cell_means.shape)
        parts.append(f"The optimal observed combination is <strong>{a_labels[best_idx[0]]} + {b_labels[best_idx[1]]}</strong> with mean {response_name} = <strong>{cell_means[best_idx]:.3f}</strong>.")

        if not replicated:
            parts.append("<em>Note: With one observation per cell there is no within-cell error df \u2014 main effects are tested against the interaction term as the error.</em>")

        return jsonify({
            "anova_table": anova_table,
            "cell_means": cell_means.tolist(),
            "a_means": a_means.tolist(),
            "b_means": b_means.tolist(),
            "chart_interaction": json.dumps(fig_interaction, cls=plotly.utils.PlotlyJSONEncoder),
            "chart_main": json.dumps(fig_main, cls=plotly.utils.PlotlyJSONEncoder),
            "interpretation": " ".join(parts),
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400
