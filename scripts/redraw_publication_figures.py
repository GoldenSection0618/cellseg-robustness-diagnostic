#!/usr/bin/env python
"""Redraw publication figures, with compact failure-diagnostic comparisons."""

from __future__ import annotations

try:
    from . import _redraw_publication_figures_core as _core
except ImportError:
    import _redraw_publication_figures_core as _core

for _name in dir(_core):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_core, _name)


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
        if max_count <= 8:
            tick_step = 2
        elif max_count <= 25:
            tick_step = 5
        elif max_count <= 50:
            tick_step = 10
        elif max_count <= 100:
            tick_step = 20
        else:
            tick_step = 50
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
            ax.barh(
                y + offset,
                values,
                height=bar_height * 0.88,
                color=method_color(method),
                label=METHOD_LABELS.get(method, method),
            )
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


def redraw_clean20_diagnostics() -> None:
    deltas = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_image_deltas.csv")
    failure_cases = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_failure_cases.csv")
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    methods = [method for method in METHOD_ORDER if method in set(non_clean["method"])]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), gridspec_kw={"width_ratios": [1.0, 1.25]})

    worst_by_image = (
        non_clean.groupby(["method", "image_id"], as_index=False)["absolute_object_f1_drop"]
        .max()
        .pivot(index="method", columns="image_id", values="absolute_object_f1_drop")
        .reindex(methods)
    )
    image_order = worst_by_image.max(axis=0).sort_values(ascending=False).index
    worst_by_image = worst_by_image[image_order]
    image = axes[0].imshow(worst_by_image.to_numpy(), aspect="auto", vmin=-0.1, vmax=1.0, cmap=DROP_CMAP)
    axes[0].set_yticks(range(len(worst_by_image.index)), labels=[METHOD_LABELS[method] for method in worst_by_image.index])
    axes[0].set_xticks(range(len(worst_by_image.columns)), labels=[str(index) for index in range(1, len(worst_by_image.columns) + 1)])
    axes[0].set_xlabel("Image rank by worst drop")
    axes[0].set_title("Worst per-image F1 drop")
    axes[0].grid(False)
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Absolute F1 drop")

    draw_failure_hint_counts(axes[1], failure_cases, methods)

    fig.suptitle("Clean20 robustness failure diagnostics", y=1.08)
    fig.tight_layout()
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
            non_clean[
                (non_clean["method"] == method) & (non_clean["perturbation"] == perturbation)
            ]["absolute_object_f1_drop"].to_numpy()
            for perturbation in non_clean_order
        ]
        box = axes[0].boxplot(
            values,
            positions=method_positions,
            widths=0.24,
            showfliers=False,
            patch_artist=True,
        )
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
            axes[0].scatter(
                method_positions[index] + jitter,
                series,
                s=7,
                alpha=0.16,
                color=method_color(method),
                linewidths=0,
            )
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


def main() -> None:
    ensure_output_dirs()
    redraw_dataset_audit()
    redraw_cellpose_method_availability()
    redraw_otsu_smoke()
    redraw_robustness_summary(
        RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_smoke_summary.csv",
        "robustness_pow_smoke",
    )
    redraw_robustness_summary(
        RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_summary.csv",
        "robustness_pow_clean20",
    )
    redraw_robustness_summary(
        RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_summary.csv",
        "robustness_pow_full_train",
    )
    redraw_yolo_threshold()
    redraw_sam2_sensitivity()
    redraw_clean20_diagnostics()
    redraw_full_train_diagnostics()
    redraw_baseline_clean_subset()
    redraw_clean_subset_count_agreement()
    redraw_cellpose_parameter_diagnostic()
    redraw_yolo_comparison(
        RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic_val_comparison_summary.csv",
        FIGURES_DIR / "supervised_yolo_label_budget_diagnostic_comparison.png",
        {
            "Cellpose-SAM": "Cellpose-SAM",
            "YOLO label-budget full train pool": "YOLO11n full",
            "YOLO label-budget 250": "YOLO11n 250",
            "YOLO fixed-budget 100": "YOLO11n 100",
            "Otsu + watershed": "Otsu",
        },
    )
    redraw_yolo_comparison(
        RESULT_SUBDIRS["supervised"] / "yolo_capacity_diagnostic_val_comparison_summary.csv",
        FIGURES_DIR / "supervised_yolo_capacity_diagnostic_comparison.png",
        {
            "Cellpose-SAM": "Cellpose-SAM",
            "YOLO11m full train pool": "YOLO11m full",
            "YOLO11n full train pool": "YOLO11n full",
            "Otsu + watershed": "Otsu",
        },
    )
    redraw_protocol_ab_heldout_comparison()
    print("Redrew publication-style summary figures from existing CSV outputs.")


if __name__ == "__main__":
    main()
