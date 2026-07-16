# Output Contract

All analysis and experiment outputs should be reproducible from code and should use
stable output directories.

## Tabular Outputs

Write important tabular outputs to categorized subdirectories under `results/`.

Recommended naming:

```text
results/<category>/<analysis_or_protocol>_<scope>.csv
```

Examples:

```text
results/dataset/dataset_inventory.csv
results/dataset/dataset_summary.csv
results/baselines/otsu_watershed_clean_subset_metrics.csv
results/baselines/cellpose_cpsam_clean_subset_metrics.csv
results/baselines/cellpose_cpsam_parameter_diagnostic_metrics.csv
results/baselines/cellpose_cpsam_parameter_diagnostic_summary.csv
results/baselines/cellpose_cpsam_parameter_diagnostic_heldout_val_metrics.csv
results/baselines/cellpose_cpsam_parameter_diagnostic_heldout_val_summary.csv
results/baselines/cellpose_cpsam_input_mode_lock_heldout_val_metrics.csv
results/baselines/cellpose_cpsam_input_mode_lock_heldout_val_summary.csv
results/baselines/sam3_prompted_concept_clean_subset_metrics.csv
results/robustness/pow_baseline_robustness_smoke_summary.csv
results/robustness/pow_baseline_robustness_clean20_summary.csv
results/robustness/pow_baseline_robustness_clean20_image_deltas.csv
results/robustness/pow_baseline_robustness_clean20_failure_cases.csv
results/robustness/pow_baseline_robustness_full_train_summary.csv
results/robustness/pow_baseline_robustness_full_train_image_deltas.csv
results/robustness/pow_baseline_robustness_full_train_failure_cases.csv
results/robustness/pow_baseline_robustness_full_train_no_prediction_cases.csv
results/supervised/yolo_label_smoke_manifest.csv
results/supervised/yolo_label_smoke_summary.csv
results/supervised/yolo_tiny_train_smoke_metadata.csv
results/supervised/yolo_tiny_train_smoke_summary.csv
results/supervised/yolo_tiny_train_smoke_metrics.csv
results/supervised/yolo_tiny_train_smoke_eval_summary.csv
results/supervised/yolo_fixed_budget_manifest.csv
results/supervised/yolo_fixed_budget_split.csv
results/supervised/yolo_fixed_budget_summary.csv
results/supervised/yolo_fixed_budget_train_metadata.csv
results/supervised/yolo_fixed_budget_train_summary.csv
results/supervised/yolo_fixed_budget_metrics.csv
results/supervised/yolo_fixed_budget_eval_summary.csv
results/supervised/yolo_fixed_budget_val_comparison_metrics.csv
results/supervised/yolo_fixed_budget_val_comparison_summary.csv
results/supervised/yolo_threshold_diagnostic_metrics.csv
results/supervised/yolo_threshold_diagnostic_summary.csv
results/supervised/yolo_label_budget_diagnostic_manifest.csv
results/supervised/yolo_label_budget_diagnostic_split.csv
results/supervised/yolo_label_budget_diagnostic_summary.csv
results/supervised/yolo_label_budget_diagnostic/budget_250/
results/supervised/yolo_label_budget_diagnostic/full_train_pool/
results/supervised/yolo_label_budget_diagnostic_budget_250_train_metadata.csv
results/supervised/yolo_label_budget_diagnostic_budget_250_train_summary.csv
results/supervised/yolo_label_budget_diagnostic_budget_250_metrics.csv
results/supervised/yolo_label_budget_diagnostic_budget_250_eval_summary.csv
results/supervised/yolo_label_budget_diagnostic_full_train_pool_train_metadata.csv
results/supervised/yolo_label_budget_diagnostic_full_train_pool_train_summary.csv
results/supervised/yolo_label_budget_diagnostic_full_train_pool_metrics.csv
results/supervised/yolo_label_budget_diagnostic_full_train_pool_eval_summary.csv
results/supervised/yolo_label_budget_diagnostic_val_comparison_metrics.csv
results/supervised/yolo_label_budget_diagnostic_val_comparison_summary.csv
results/supervised/yolo_capacity_diagnostic_yolo11m_train_metadata.csv
results/supervised/yolo_capacity_diagnostic_yolo11m_train_summary.csv
results/supervised/yolo_capacity_diagnostic_yolo11m_metrics.csv
results/supervised/yolo_capacity_diagnostic_yolo11m_eval_summary.csv
results/supervised/yolo_capacity_diagnostic_val_comparison_metrics.csv
results/supervised/yolo_capacity_diagnostic_val_comparison_summary.csv
```

Initial result categories:

- `results/dataset/`
- `results/baselines/`
- `results/robustness/`
- `results/supervised/`
- `results/vlm/`

CSV files should include identifiers that make rows traceable back to source data,
such as:

- `split`
- `image_id`
- `image_path`
- `method`
- `perturbation`
- `parameter_set`

Grouped metric summaries should include the standard object metrics plus the
failure-rate diagnostics when the source metrics contain instance counts:

- `mean_missed_object_rate`
- `mean_fp_per_true_instance`
- `mean_count_bias`

## Figure Outputs

Write generated figures to `figures/`.

Recommended naming:

```text
figures/<analysis_or_protocol>_<description>.png
```

Examples:

```text
figures/dataset_split_counts.png
figures/dataset_train_instance_count_hist.png
figures/otsu_watershed_subset_overlay_examples.png
figures/cellpose_cpsam_parameter_diagnostic_f1.png
figures/cellpose_cpsam_parameter_diagnostic_heldout_val_f1.png
figures/sam3_prompted_concept_clean_subset_overlay_examples.png
figures/robustness_pow_smoke_mean_f1.png
figures/robustness_pow_smoke_relative_f1_drop.png
figures/robustness_pow_clean20_summary.png
figures/robustness_pow_clean20_failure_diagnostics.png
figures/robustness_pow_full_train_summary.png
figures/robustness_pow_full_train_failure_diagnostics.png
figures/robustness_sam2_amg_sensitivity_clean20_mean_f1.png
figures/supplementary_baseline_clean_subset_image_method_f1_heatmap.png
figures/supervised_yolo_label_smoke_overlays.png
figures/supervised_yolo_tiny_train_smoke_overlays.png
figures/supervised_yolo_fixed_budget_overlays.png
figures/supervised_yolo_fixed_budget_eval_overlays.png
figures/supervised_yolo_threshold_diagnostic_f1.png
figures/supervised_yolo_threshold_diagnostic_count_error.png
figures/supervised_yolo_label_budget_diagnostic_comparison.png
figures/supervised_yolo_label_budget_diagnostic_budget_250_eval_overlays.png
figures/supervised_yolo_label_budget_diagnostic_full_train_pool_eval_overlays.png
figures/supervised_yolo_capacity_diagnostic_yolo11m_eval_overlays.png
figures/supervised_yolo_capacity_diagnostic_comparison.png
```

Figures should be generated by scripts rather than manually edited. Keep figures in
the flat `figures/` directory and use clear filenames, titles, axis labels, and
captions encoded in plotting code.

Use heatmaps only when the row-by-column matrix structure is itself informative,
such as shared difficult images or configuration-by-perturbation sensitivity. Use
position- or length-based plots for small method comparisons and exact differences.
Supplementary heatmaps should be prefixed with `supplementary_`.

Metric, summary, audit, comparison, and diagnostic figures are rendered only by
`scripts/redraw_publication_figures.py`, which reads existing CSV outputs and does
not rerun experiments. Experiment and evaluation scripts should write CSV outputs;
they may write figure files only for segmentation overlay examples that visually
check label or prediction alignment.

Use `cellseg_robustness.plot_style.save_png` for final figure export. Figures
should be PNG only, with the shared publication-style matplotlib defaults applied
through `cellseg_robustness.plot_style`.
