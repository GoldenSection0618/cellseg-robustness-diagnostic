#!/usr/bin/env python
"""Publication-focused plotting overrides for selected diagnostic figures."""

from __future__ import annotations

try:
    from . import _redraw_publication_figures_core as core
except ImportError:
    import _redraw_publication_figures_core as core

np = core.np
pd = core.pd
plt = core.plt
Line2D = core.Line2D

FIGURES_DIR = core.FIGURES_DIR
RESULT_SUBDIRS = core.RESULT_SUBDIRS
METHOD_ORDER = core.METHOD_ORDER
METHOD_LABELS = core.METHOD_LABELS
BASELINE_METHOD_LABELS = core.BASELINE_METHOD_LABELS
PERTURBATION_LABELS = core.PERTURBATION_LABELS
PERTURBATION_ORDER = core.PERTURBATION_ORDER
SCORE_CMAP = core.SCORE_CMAP
DROP_CMAP = core.DROP_CMAP

method_color = core.method_color
method_marker = core.method_marker
style_axis = core.style_axis
save_png = core.save_png
gradient_colors = core.gradient_colors


def _sam2_config_label(config_id: str) -> str:
    """Convert a SAM2 sweep identifier into a compact, meaningful label."""
    exact = {"default_current": "Default"}
    if config_id in exact:
        return exact[config_id]

    prefixes = [
        ("points_per_side_", "PPS "),
        ("stability_score_thresh_", "Stability "),
        ("pred_iou_thresh_", "Pred IoU "),
        ("box_nms_thresh_", "Box NMS "),
        ("crop_n_layers_", "Crop layers "),
        ("min_mask_region_area_", "Min area "),
    ]
    for prefix, label in prefixes:
        if config_id.startswith(prefix):
            return label + config_id.removeprefix(prefix)
    return config_id.replace("_", " ")


def redraw_robustness_summary(summary_path, figure_prefix: str) -> None:
    """Keep the smoke-stage line plots and omit the redundant small heatmap."""
    if figure_prefix != "robustness_pow_smoke":
        core.redraw_robustness_summary(summary_path, figure_prefix)
        return

    summary = pd.read_csv(summary_path)
    summary = summary[summary["perturbation"].isin(PERTURBATION_ORDER)].copy()
    methods = [method for method in METHOD_ORDER if method in set(summary["method"])]

    pivot = (
        summary.pivot(index="perturbation", columns="method", values="mean_object_f1")
        .reindex(PERTURBATION_ORDER)
        .dropna(how="all")
    )
    pivot = pivot[[method for method in methods if method in pivot.columns]]
    labels = [PERTURBATION_LABELS[item] for item in pivot.index]
    x = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    for method in pivot.columns:
        ax.plot(
            x,
            pivot[method].to_numpy(),
            marker=method_marker(method),
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.0,
            label=METHOD_LABELS.get(method, method),
            color=method_color(method),
        )
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean object F1")
    ax.set_title("Smoke-stage robustness across perturbations")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=min(len(pivot.columns), 3))
    style_axis(ax, grid_axis="y")
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_mean_f1.png")
    plt.close(fig)

    drop = (
        summary[summary["perturbation"] != "clean"]
        .pivot(index="perturbation", columns="method", values="relative_object_f1_drop")
        .reindex([item for item in PERTURBATION_ORDER if item != "clean"])
    )
    drop = drop[[method for method in methods if method in drop.columns]].dropna(how="all")
    x = np.arange(len(drop))

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    for method in drop.columns:
        values = drop[method].to_numpy()
        ax.plot(
            x,
            values,
            marker=method_marker(method),
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.0,
            label=METHOD_LABELS.get(method, method),
            color=method_color(method),
        )
        if np.isfinite(values).any():
            max_index = int(np.nanargmax(values))
            max_value = values[max_index]
            if max_value >= 0.15:
                ax.annotate(
                    f"{max_value:.0%}",
                    xy=(x[max_index], max_value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#374151",
                    arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.7},
                )
    ax.axhline(0, color="#4b5563", linewidth=0.8)
    ax.set_xticks(x, labels=[PERTURBATION_LABELS[item] for item in drop.index])
    ax.set_ylabel("Relative F1 drop")
    ax.set_title("Smoke-stage performance loss from clean")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=min(len(drop.columns), 3))
    style_axis(ax, grid_axis="y")
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_relative_f1_drop.png")
    plt.close(fig)

    obsolete = FIGURES_DIR / f"{figure_prefix}_method_condition_heatmap.png"
    if obsolete.exists():
        obsolete.unlink()


def redraw_sam2_sensitivity() -> None:
    """Render SAM2 sensitivity with meaningful labels and robustness-based ordering."""
    clean = pd.read_csv(RESULT_SUBDIRS["robustness"] / "sam2_amg_sensitivity_clean20_clean_screen_summary.csv")
    validation = pd.read_csv(RESULT_SUBDIRS["robustness"] / "sam2_amg_sensitivity_clean20_validation_summary.csv")

    clean = clean.copy()
    clean["label"] = clean["config_id"].map(_sam2_config_label)
    clean = clean.sort_values("mean_object_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.barh(clean["label"], clean["mean_object_f1"], color=gradient_colors(len(clean)))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean object F1")
    ax.set_title("SAM2 AMG clean-screen sensitivity")
    style_axis(ax, grid_axis="x")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_clean_screen_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.barh(clean["label"], clean["mean_pred_instances"], color=gradient_colors(len(clean)))
    ax.axvline(clean["mean_true_instances"].mean(), color="#4b5563", linewidth=0.8, linestyle="--", label="Mean true count")
    ax.set_xlabel("Mean predicted instances")
    ax.set_title("SAM2 AMG clean-screen counts")
    ax.legend(loc="lower right")
    style_axis(ax, grid_axis="x")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_clean_screen_counts.png")
    plt.close(fig)

    clean_scores = (
        validation[validation["perturbation"] == "clean"]
        .set_index("config_id")["mean_object_f1"]
        .rename("clean_f1")
    )
    perturbed = validation[validation["perturbation"] != "clean"]
    robustness = perturbed.groupby("config_id")["mean_object_f1"].agg(
        mean_perturbed_f1="mean",
        worst_perturbed_f1="min",
    )
    ranking = robustness.join(clean_scores, how="left").sort_values(
        ["mean_perturbed_f1", "worst_perturbed_f1", "clean_f1"],
        ascending=False,
    )
    selected_ids = ranking.head(6).index.tolist()
    label_map = {config_id: _sam2_config_label(config_id) for config_id in selected_ids}

    frame = validation[validation["config_id"].isin(selected_ids)].copy()
    frame["label"] = frame["config_id"].map(label_map)
    pivot = frame.pivot(index="label", columns="perturbation", values="mean_object_f1")
    pivot = pivot[[item for item in PERTURBATION_ORDER if item in pivot.columns]]
    ordered_labels = [label_map[config_id] for config_id in selected_ids]
    pivot = pivot.reindex(ordered_labels)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap=SCORE_CMAP, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=[PERTURBATION_LABELS[item] for item in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("SAM2 AMG configuration sensitivity\nsorted by mean perturbed F1")
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iat[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 0.55 else "#111827",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean object F1")
    ax.grid(False)
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_mean_f1.png")
    plt.close(fig)

    count_error = (
        frame.groupby("config_id", as_index=False)["mean_absolute_count_error"]
        .mean()
        .assign(label=lambda df: df["config_id"].map(label_map))
    )
    rank_map = {config_id: index for index, config_id in enumerate(selected_ids)}
    count_error["rank"] = count_error["config_id"].map(rank_map)
    count_error = count_error.sort_values("rank")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(count_error["label"], count_error["mean_absolute_count_error"], color=gradient_colors(len(count_error)))
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute count error")
    ax.set_title("SAM2 AMG count error\nrobustness-ranked configurations")
    style_axis(ax, grid_axis="x")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_count_error.png")
    plt.close(fig)


def redraw_baseline_clean_subset() -> None:
    """Run the core baseline redraw, then replace its heatmap with a supplementary diagnostic."""
    core.redraw_baseline_clean_subset()

    baseline_files = {
        "otsu_watershed": "otsu_watershed_clean_subset_metrics.csv",
        "cellpose_cpsam": "cellpose_cpsam_clean_subset_metrics.csv",
        "sam2_amg": "sam2_amg_clean_subset_metrics.csv",
    }
    frames = []
    for method, filename in baseline_files.items():
        frame = pd.read_csv(RESULT_SUBDIRS["baselines"] / filename)
        frame["method"] = method
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)

    pivot = metrics.pivot(index="image_id", columns="method", values="object_f1")[METHOD_ORDER]
    disagreement = pivot.max(axis=1) - pivot.min(axis=1)
    pivot = pivot.loc[disagreement.sort_values(ascending=False).index]
    rank_labels = [str(index) for index in range(1, len(pivot) + 1)]

    fig, ax = plt.subplots(figsize=(5.8, 6.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=SCORE_CMAP)
    ax.set_xticks(range(len(METHOD_ORDER)), labels=[BASELINE_METHOD_LABELS[method] for method in METHOD_ORDER])
    tick_positions = list(range(len(rank_labels))) if len(rank_labels) <= 25 else list(range(0, len(rank_labels), 5))
    ax.set_yticks(tick_positions, labels=[rank_labels[index] for index in tick_positions])
    ax.set_title("Supplementary: per-image object F1 by method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Image rank by cross-method F1 range")
    if pivot.shape[0] <= 30:
        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.iat[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.65 else "#111827",
                    fontsize=6,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Object F1")
    ax.grid(False)
    save_png(fig, FIGURES_DIR / "supplementary_baseline_clean_subset_image_method_f1_heatmap.png")
    plt.close(fig)

    obsolete = FIGURES_DIR / "baseline_clean_subset_image_method_f1_heatmap.png"
    if obsolete.exists():
        obsolete.unlink()


def draw_failure_hint_counts(ax, failure_cases: pd.DataFrame, methods: list[str]) -> None:
    """Draw directly comparable failure-hint counts for two or more methods."""
    counts = (
        failure_cases.groupby(["method", "failure_hint"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(methods)
        .fillna(0)
        .astype(int)
    )
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    if counts.empty:
        ax.axis("off")
        return

    preferred_order = ["NO_PRED", "COLLAPSE", "FN+FP", "FN", "FP/OVER", "COUNT", "MIXED", "NO_DROP"]
    available = [hint for hint in preferred_order if hint in counts.columns]
    available.extend(hint for hint in counts.columns if hint not in available)
    counts = counts[available]
    hint_order = counts.sum(axis=0).sort_values(ascending=False).index.tolist()
    counts = counts[hint_order]

    y = np.arange(len(hint_order))
    max_count = max(int(counts.to_numpy().max()), 1)
    label_pad = max(0.45, max_count * 0.025)

    if len(methods) == 2:
        left_method, right_method = methods
        left_values = counts.loc[left_method].to_numpy(dtype=float)
        right_values = counts.loc[right_method].to_numpy(dtype=float)
        ax.barh(y, -left_values, height=0.62, color=method_color(left_method))
        ax.barh(y, right_values, height=0.62, color=method_color(right_method))
        ax.axvline(0, color="#111827", linewidth=1.0, zorder=3)

        for ypos, value in zip(y, left_values):
            xpos = -value - label_pad if value > 0 else -label_pad
            ax.text(xpos, ypos, f"{int(value)}", ha="right", va="center", fontsize=8, color="#374151")
        for ypos, value in zip(y, right_values):
            xpos = value + label_pad if value > 0 else label_pad
            ax.text(xpos, ypos, f"{int(value)}", ha="left", va="center", fontsize=8, color="#374151")

        axis_limit = max_count + max(2, max_count * 0.12)
        tick_step = 2 if max_count <= 8 else 5 if max_count <= 25 else 10 if max_count <= 50 else 20 if max_count <= 100 else 50
        tick_max = int(np.ceil(axis_limit / tick_step) * tick_step)
        positive_ticks = np.arange(0, tick_max + tick_step, tick_step)
        ticks = np.concatenate((-positive_ticks[:0:-1], positive_ticks))
        ax.set_xlim(-tick_max, tick_max)
        ax.set_xticks(ticks, labels=[str(abs(int(value))) for value in ticks])
        ax.text(0.25, 1.02, METHOD_LABELS.get(left_method, left_method), transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        ax.text(0.75, 1.02, METHOD_LABELS.get(right_method, right_method), transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    else:
        group_height = 0.78
        bar_height = group_height / max(len(methods), 1)
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * bar_height
        for offset, method in zip(offsets, methods):
            values = counts.loc[method].to_numpy(dtype=float)
            ax.barh(y + offset, values, height=bar_height * 0.88, color=method_color(method), label=METHOD_LABELS.get(method, method))
            for ypos, value in zip(y + offset, values):
                if value > 0:
                    ax.text(value + label_pad, ypos, f"{int(value)}", ha="left", va="center", fontsize=7.5, color="#374151")
        ax.set_xlim(0, max_count * 1.22 + 1)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=len(methods))

    ax.set_yticks(y, labels=hint_order)
    ax.invert_yaxis()
    ax.set_xlabel("Worst-case row count")
    style_axis(ax, grid_axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_sam2_failure_strip(ax, failure_cases: pd.DataFrame, method: str = "sam2_amg") -> None:
    """Summarize the collapse-dominated SAM2 AMG failure regime."""
    counts = failure_cases.loc[failure_cases["method"] == method, "failure_hint"].value_counts().astype(int)
    if counts.empty:
        ax.axis("off")
        return

    preferred_order = ["COLLAPSE", "FN+FP", "NO_PRED", "FN", "FP/OVER", "COUNT", "MIXED", "NO_DROP"]
    hint_order = [hint for hint in preferred_order if counts.get(hint, 0) > 0]
    hint_order.extend(hint for hint in counts.index if hint not in hint_order)
    total = int(counts.sum())
    color = method_color(method)
    left = 0.0

    for hint in hint_order:
        count = int(counts[hint])
        share = 100.0 * count / total
        is_collapse = hint == "COLLAPSE"
        ax.barh([0], [share], left=left, height=0.52, color=color, alpha=0.92 if is_collapse else 0.28, edgecolor=color, linewidth=0.9, hatch=None if is_collapse else "///")
        label = f"{hint} {count} ({share:.0f}%)"
        center = left + share / 2
        if share >= 24:
            ax.text(center, 0, label, ha="center", va="center", fontsize=8, color="white" if is_collapse else "#111827", fontweight="bold" if is_collapse else "normal")
        else:
            ax.annotate(label, xy=(center, 0.26), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color="#374151", arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.7})
        left += share

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.48, 0.78)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100], labels=["0", "50", "100"])
    ax.set_xlabel("Share of SAM2 AMG worst-case rows (%)", fontsize=8)
    ax.set_title("SAM2 AMG failure regime", loc="left", fontsize=9.5, pad=4)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#111827")
    ax.tick_params(axis="x", labelsize=8, colors="#111827")


def redraw_clean20_diagnostics() -> None:
    deltas = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_image_deltas.csv")
    failure_cases = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_failure_cases.csv")
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    methods = [method for method in METHOD_ORDER if method in set(non_clean["method"])]
    pair_methods = [method for method in ["otsu_watershed", "cellpose_cpsam"] if method in methods]
    sam2_method = "sam2_amg" if "sam2_amg" in methods else None

    fig = plt.figure(figsize=(12.0, 5.0), constrained_layout=True)
    outer_grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.28])
    heatmap_ax = fig.add_subplot(outer_grid[0])

    if sam2_method is not None:
        right_grid = outer_grid[1].subgridspec(2, 1, height_ratios=[4.6, 1.15], hspace=0.34)
        pair_ax = fig.add_subplot(right_grid[0])
        sam2_ax = fig.add_subplot(right_grid[1])
    else:
        pair_ax = fig.add_subplot(outer_grid[1])
        sam2_ax = None

    worst_by_image = (
        non_clean.groupby(["method", "image_id"], as_index=False)["absolute_object_f1_drop"]
        .max()
        .pivot(index="method", columns="image_id", values="absolute_object_f1_drop")
        .reindex(methods)
    )
    image_order = worst_by_image.max(axis=0).sort_values(ascending=False).index
    worst_by_image = worst_by_image[image_order]
    image = heatmap_ax.imshow(worst_by_image.to_numpy(), aspect="auto", vmin=-0.1, vmax=1.0, cmap=DROP_CMAP)
    heatmap_ax.set_yticks(range(len(worst_by_image.index)), labels=[METHOD_LABELS[method] for method in worst_by_image.index])
    heatmap_ax.set_xticks(range(len(worst_by_image.columns)), labels=[str(index) for index in range(1, len(worst_by_image.columns) + 1)])
    heatmap_ax.set_xlabel("Image rank (joint max-drop order)")
    heatmap_ax.set_title("Shared worst-case F1 drops by image")
    heatmap_ax.grid(False)
    colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Absolute F1 drop")

    draw_failure_hint_counts(pair_ax, failure_cases, pair_methods)
    if sam2_ax is not None and sam2_method is not None:
        draw_sam2_failure_strip(sam2_ax, failure_cases, sam2_method)

    fig.suptitle("Clean20 robustness failure diagnostics", y=1.04)
    save_png(fig, FIGURES_DIR / "robustness_pow_clean20_failure_diagnostics.png")
    plt.close(fig)


def redraw_full_train_diagnostics() -> None:
    deltas = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_image_deltas.csv")
    failure_cases = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_failure_cases.csv")
    methods = [method for method in ["otsu_watershed", "cellpose_cpsam"] if method in set(deltas["method"])]
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    non_clean_order = [name for name in PERTURBATION_ORDER if name != "clean"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.05, 1.2]})
    positions = np.arange(len(non_clean_order))
    offset = 0.16
    for method_index, method in enumerate(methods):
        method_positions = positions + (method_index - (len(methods) - 1) / 2) * offset * 2
        values = [
            non_clean[(non_clean["method"] == method) & (non_clean["perturbation"] == perturbation)]["absolute_object_f1_drop"].to_numpy()
            for perturbation in non_clean_order
        ]
        box = axes[0].boxplot(values, positions=method_positions, widths=0.24, showfliers=False, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor(method_color(method))
            patch.set_alpha(0.18)
            patch.set_edgecolor(method_color(method))
        for median in box["medians"]:
            median.set_color(method_color(method))
            median.set_linewidth(1.6)
        for index, series in enumerate(values):
            if len(series) == 0:
                continue
            jitter = ((np.arange(len(series)) % 9) - 4) * 0.007
            axes[0].scatter(method_positions[index] + jitter, series, s=7, alpha=0.16, color=method_color(method), linewidths=0)
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_xticks(positions, labels=[PERTURBATION_LABELS[label] for label in non_clean_order])
    axes[0].set_ylabel("Absolute object F1 drop from clean")
    axes[0].set_title("Per-image F1 drop distributions")
    style_axis(axes[0], grid_axis="y")

    draw_failure_hint_counts(axes[1], failure_cases, methods)

    legend_handles = [
        Line2D([0], [0], color=method_color(method), marker=method_marker(method), markerfacecolor="white", markeredgewidth=1.4, label=METHOD_LABELS[method])
        for method in methods
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncols=len(legend_handles), bbox_to_anchor=(0.31, 1.02), frameon=False)
    fig.suptitle("Full-train robustness failure diagnostics", y=1.12)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "robustness_pow_full_train_failure_diagnostics.png")
    plt.close(fig)
