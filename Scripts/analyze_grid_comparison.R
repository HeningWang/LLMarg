# =============================================================================
# analyze_grid_comparison.R
# Compares LLM grid-search results to human baseline data.
# Run from the Scripts/ directory or the project root.
# =============================================================================

library(tidyverse)
library(patchwork)
library(cspplot)
library(scales)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
project_root <- here::here()
if (!dir.exists(file.path(project_root, "project"))) {
  project_root <- dirname(getwd())
}

results_dir  <- file.path(project_root, "results")
output_dir   <- file.path(project_root, "results", "analyze")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

human_summary_path <- file.path(output_dir, "human_baseline_summary.csv")
human_item_path    <- file.path(output_dir, "human_item_distributions.csv")

# CSP palette
project_colors_init <- cspplot::list_colors() |> pull(hex)
cb_palette <- c(
  "high" = project_colors_init[3],   # crayola
  "low"  = project_colors_init[1],   # glaucous
  "info" = project_colors_init[4]    # fern
)

# ---------------------------------------------------------------------------
# 1. Load human baseline
# ---------------------------------------------------------------------------
if (!file.exists(human_summary_path)) {
  stop("Human baseline summary not found. Run analyze_human_baseline.R first.\n",
       "Expected path: ", human_summary_path)
}

human_summary <- read_csv(human_summary_path, show_col_types = FALSE)
human_dist <- human_summary %>%
  select(condition, response_label, proportion) %>%
  distinct()

cat("Human baseline loaded.\n")

# Load item-level data if available (for correlation analysis)
human_item <- if (file.exists(human_item_path)) {
  read_csv(human_item_path, show_col_types = FALSE)
} else {
  NULL
}

# ---------------------------------------------------------------------------
# 2. Load LLM grid results
# ---------------------------------------------------------------------------
grid_files <- Sys.glob(file.path(results_dir, "grid_*.csv"))

if (length(grid_files) == 0) {
  cat("No grid result files found (pattern: results/grid_*.csv).\n")
  cat("Run the experiment first, then re-run this script.\n")
  cat("Exiting gracefully.\n")
  quit(save = "no", status = 0)
}

cat("Found", length(grid_files), "grid result file(s):\n")
cat(paste(" -", basename(grid_files)), sep = "\n")

# Read all grid files and infer model name from filename
read_grid_file <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  # Extract model name from filename: grid_<model>_<timestamp>.csv
  fname <- basename(path)
  model_name <- fname %>%
    str_remove("^grid_") %>%
    str_remove("_\\d{8}_\\d{6}\\.csv$")
  df %>% mutate(model = model_name)
}

llm_raw <- map_dfr(grid_files, read_grid_file)
cat("\nTotal LLM observations:", nrow(llm_raw), "\n")

# Harmonize confidence column name: grid files use "confidence",
# older deepseek file uses "confidence_logprob"
if (!"confidence" %in% names(llm_raw) && "confidence_logprob" %in% names(llm_raw)) {
  llm_raw <- llm_raw %>% rename(confidence = confidence_logprob)
}

# ---------------------------------------------------------------------------
# 3. Flag invalid responses
# ---------------------------------------------------------------------------
llm_raw <- llm_raw %>%
  mutate(
    is_invalid = str_starts(chosen_speaker, "Invalid") |
                 str_starts(chosen_speaker, "ERROR") |
                 is.na(chosen_speaker)
  )

invalid_summary <- llm_raw %>%
  group_by(model, prompt_template) %>%
  summarise(
    total     = n(),
    n_invalid = sum(is_invalid),
    invalid_rate = mean(is_invalid),
    .groups = "drop"
  ) %>%
  arrange(desc(invalid_rate))

cat("\n--- Invalid response rates ---\n")
print(invalid_summary)

# Keep only valid responses for distribution analysis
llm_valid <- llm_raw %>% filter(!is_invalid)

if (nrow(llm_valid) == 0) {
  cat("\nAll LLM responses are invalid. No comparison possible.\n")
  cat("Saving invalid summary and exiting.\n")
  write_csv(invalid_summary, file.path(output_dir, "grid_invalid_rates.csv"))
  quit(save = "no", status = 0)
}

# ---------------------------------------------------------------------------
# 4. Compute LLM response distributions per grid cell
# ---------------------------------------------------------------------------
# A grid cell = model x prompt_template x temperature
llm_dist <- llm_valid %>%
  count(model, prompt_template, temperature, condition, chosen_speaker) %>%
  group_by(model, prompt_template, temperature, condition) %>%
  mutate(
    total      = sum(n),
    proportion = n / total
  ) %>%
  ungroup() %>%
  rename(response_label = chosen_speaker)

# Ensure all response categories are present (fill missing with 0)
all_combos <- llm_dist %>%
  distinct(model, prompt_template, temperature, condition) %>%
  crossing(response_label = c("high", "low", "info"))

llm_dist <- all_combos %>%
  left_join(llm_dist, by = c("model", "prompt_template", "temperature",
                              "condition", "response_label")) %>%
  mutate(
    n          = replace_na(n, 0L),
    proportion = replace_na(proportion, 0)
  ) %>%
  group_by(model, prompt_template, temperature, condition) %>%
  mutate(total = sum(n)) %>%
  ungroup()

# ---------------------------------------------------------------------------
# 5. Match rates per grid cell
# ---------------------------------------------------------------------------
llm_match <- llm_valid %>%
  mutate(is_match = (chosen_speaker == condition)) %>%
  group_by(model, prompt_template, temperature, condition) %>%
  summarise(
    n_trials   = n(),
    n_match    = sum(is_match),
    match_rate = mean(is_match),
    .groups    = "drop"
  )

# ---------------------------------------------------------------------------
# 6. Jensen-Shannon divergence & total variation distance
# ---------------------------------------------------------------------------
# JS divergence (base 2)
js_divergence <- function(p, q) {
  # Handle zero probabilities with small epsilon
  eps <- 1e-10
  p <- pmax(p, eps)
  q <- pmax(q, eps)
  # Normalize

  p <- p / sum(p)
  q <- q / sum(q)
  m <- (p + q) / 2
  kl_pm <- sum(p * log2(p / m))
  kl_qm <- sum(q * log2(q / m))
  (kl_pm + kl_qm) / 2
}

# Total variation distance
tv_distance <- function(p, q) {
  sum(abs(p - q)) / 2
}

# Ensure human_dist has all three response labels per condition
human_dist_full <- crossing(
  condition = c("high", "low", "info"),
  response_label = c("high", "low", "info")
) %>%
  left_join(human_dist, by = c("condition", "response_label")) %>%
  mutate(proportion = replace_na(proportion, 0))

# Compute divergence metrics for each grid cell x condition
divergence_metrics <- llm_dist %>%
  distinct(model, prompt_template, temperature, condition) %>%
  pmap_dfr(function(model, prompt_template, temperature, condition) {
    llm_p <- llm_dist %>%
      filter(.data$model == .env$model,
             .data$prompt_template == .env$prompt_template,
             .data$temperature == .env$temperature,
             .data$condition == .env$condition) %>%
      arrange(response_label) %>%
      pull(proportion)

    human_p <- human_dist_full %>%
      filter(.data$condition == .env$condition) %>%
      arrange(response_label) %>%
      pull(proportion)

    tibble(
      model           = model,
      prompt_template = prompt_template,
      temperature     = temperature,
      condition       = condition,
      js_divergence   = js_divergence(llm_p, human_p),
      tv_distance     = tv_distance(llm_p, human_p)
    )
  })

# ---------------------------------------------------------------------------
# 7. Item-level correlation (if item data available)
# ---------------------------------------------------------------------------
item_correlation <- NULL
if (!is.null(human_item)) {
  # For each grid cell, compute correlation between LLM and human
  # item-level match rates
  item_correlation <- llm_valid %>%
    mutate(is_match = (chosen_speaker == condition)) %>%
    group_by(model, prompt_template, temperature, itemID, condition) %>%
    summarise(llm_match_rate = mean(is_match), .groups = "drop") %>%
    inner_join(
      human_item %>%
        filter(response_label == condition) %>%
        select(itemID, condition, human_prop = proportion),
      by = c("itemID", "condition")
    ) %>%
    group_by(model, prompt_template, temperature) %>%
    summarise(
      item_correlation = cor(llm_match_rate, human_prop, use = "complete.obs"),
      n_items          = n(),
      .groups          = "drop"
    )
}

# ---------------------------------------------------------------------------
# 8. Combine and save comparison metrics
# ---------------------------------------------------------------------------
# Aggregate divergence across conditions
grid_summary <- divergence_metrics %>%
  group_by(model, prompt_template, temperature) %>%
  summarise(
    mean_js_divergence = mean(js_divergence),
    mean_tv_distance   = mean(tv_distance),
    .groups = "drop"
  )

# Add per-condition match rates (wide format)
match_wide <- llm_match %>%
  select(model, prompt_template, temperature, condition, match_rate) %>%
  pivot_wider(
    names_from  = condition,
    values_from = match_rate,
    names_prefix = "match_"
  )

grid_summary <- grid_summary %>%
  left_join(match_wide, by = c("model", "prompt_template", "temperature"))

# Add item correlation if available
if (!is.null(item_correlation)) {
  grid_summary <- grid_summary %>%
    left_join(item_correlation, by = c("model", "prompt_template", "temperature"))
}

# Add invalid rate
grid_summary <- grid_summary %>%
  left_join(
    invalid_summary %>% select(model, prompt_template, invalid_rate),
    by = c("model", "prompt_template")
  )

# Rank by mean JS divergence (lower = closer to human)
grid_summary <- grid_summary %>%
  arrange(mean_js_divergence)

write_csv(grid_summary, file.path(output_dir, "grid_comparison_metrics.csv"))
cat("\nSaved comparison metrics to:", file.path(output_dir, "grid_comparison_metrics.csv"), "\n")

# Save per-condition divergence detail
write_csv(divergence_metrics, file.path(output_dir, "grid_divergence_detail.csv"))

# Save invalid rates
write_csv(invalid_summary, file.path(output_dir, "grid_invalid_rates.csv"))

cat("\n--- Grid summary (ranked by closeness to human data) ---\n")
print(grid_summary)

# ---------------------------------------------------------------------------
# 9. Visualization: side-by-side human vs LLM distributions
# ---------------------------------------------------------------------------

# Bootstrap CIs for LLM distributions (using runs as resampling units)
set.seed(42)
n_boot <- 1000

llm_boot_ci <- llm_valid %>%
  group_by(model, prompt_template, temperature, condition, chosen_speaker) %>%
  group_modify(function(.data, .key) {
    cell_data <- llm_valid %>%
      filter(
        model           == .key$model,
        prompt_template == .key$prompt_template,
        temperature     == .key$temperature,
        condition       == .key$condition
      )
    if (nrow(cell_data) < 2) {
      return(tibble(ci_lower = NA_real_, ci_upper = NA_real_))
    }
    boot_props <- replicate(n_boot, {
      boot_sample <- slice_sample(cell_data, n = nrow(cell_data), replace = TRUE)
      mean(boot_sample$chosen_speaker == .key$chosen_speaker)
    })
    tibble(
      ci_lower = quantile(boot_props, 0.025, na.rm = TRUE),
      ci_upper = quantile(boot_props, 0.975, na.rm = TRUE)
    )
  }) %>%
  ungroup() %>%
  rename(response_label = chosen_speaker)

llm_plot_data <- llm_dist %>%
  left_join(llm_boot_ci, by = c("model", "prompt_template", "temperature",
                                 "condition", "response_label"))

# Human plot data (reuse from summary)
human_plot_data <- human_dist_full %>%
  mutate(source = "Human")

# ---------------------------------------------------------------------------
# 9. MAIN RESULT PLOT
#    x-axis: 3 conditions (high, low, info)
#    Bars: Human + 5 probing methods side by side (match rate)
#    Faceted by model
#    With bootstrapped 95% CIs
# ---------------------------------------------------------------------------
theme_set(theme_csp())
project_colors <- cspplot::list_colors() |> pull(hex)

csp_palette <- c(
  "high" = project_colors[3],   # crayola
  "low"  = project_colors[1],   # glaucous
  "info" = project_colors[4]    # fern
)

# Compute match rate per model x method x condition with bootstrap CIs
set.seed(42)
n_boot_llm <- 1000

llm_match_ci <- llm_valid %>%
  group_by(model, prompt_template, condition) %>%
  group_modify(function(.data, .key) {
    is_match <- .data$chosen_speaker == .key$condition
    if (length(is_match) < 2) {
      return(tibble(match_rate = mean(is_match), ci_lower = NA_real_,
                    ci_upper = NA_real_, n = length(is_match)))
    }
    boot_rates <- replicate(n_boot_llm, mean(sample(is_match, replace = TRUE)))
    tibble(
      match_rate = mean(is_match),
      ci_lower   = quantile(boot_rates, 0.025, na.rm = TRUE),
      ci_upper   = quantile(boot_rates, 0.975, na.rm = TRUE),
      n          = length(is_match)
    )
  }) %>%
  ungroup() %>%
  mutate(source = prompt_template)

# Human match rates with CIs (recompute from raw data if needed)
human_match_ci <- human_dist_full %>%
  filter(response_label == condition) %>%
  mutate(
    match_rate = proportion,
    source     = "Human"
  ) %>%
  select(condition, match_rate, source)

# Compute human match rate CIs from raw data
human_raw_path <- file.path(project_root, "project", "data", "data_listenerside.csv")
human_raw <- read_csv(human_raw_path, show_col_types = FALSE) %>%
  filter(!str_starts(condition, "sample\\."), !is.na(response)) %>%
  mutate(
    response_label = case_when(
      response == "Teacher"  ~ "high",
      response == "Student"  ~ "low",
      response == "Examiner" ~ "info"
    ),
    is_match = (response_label == condition)
  )

human_ci_for_main <- human_raw %>%
  group_by(condition) %>%
  summarise(
    match_rate = mean(is_match),
    ci_lower   = quantile(replicate(1000, mean(sample(is_match, replace = TRUE))), 0.025),
    ci_upper   = quantile(replicate(1000, mean(sample(is_match, replace = TRUE))), 0.975),
    .groups = "drop"
  ) %>%
  mutate(source = "Human")

# Combine
main_plot_data <- bind_rows(
  human_ci_for_main %>%
    crossing(model = unique(llm_match_ci$model)),
  llm_match_ci %>%
    select(model, condition, match_rate, ci_lower, ci_upper, source)
) %>%
  mutate(
    condition = factor(condition, levels = c("high", "low", "info")),
    source    = factor(source, levels = c(
      "Human", "direct_label", "meta_label",
      "chain_of_thought", "self_reported_confidence", "log_probs"
    ))
  )

# Nice labels for methods
method_labels <- c(
  "Human"                    = "Human",
  "direct_label"             = "Direct label",
  "meta_label"               = "Meta label",
  "chain_of_thought"         = "Chain of thought",
  "self_reported_confidence" = "Self-rep. confidence",
  "log_probs"                = "Log probs"
)

# 6 colors: human + 5 methods
main_palette <- c(
  "Human"                    = project_colors[6],   # independence
  "direct_label"             = project_colors[3],   # crayola
  "meta_label"               = project_colors[10],  # crayola dark
  "chain_of_thought"         = project_colors[1],   # glaucous
  "self_reported_confidence" = project_colors[2],   # shimmer
  "log_probs"                = project_colors[5]    # opal
)

p_main <- ggplot(main_plot_data, aes(
  x = condition, y = match_rate, fill = source
)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.8),
    width = 0.2, linewidth = 0.3
  ) +
  geom_hline(yintercept = 1 / 3, linetype = "dashed", colour = "grey40",
             linewidth = 0.3) +
  scale_fill_manual(values = main_palette, labels = method_labels, name = "Source") +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  facet_wrap(~ model, nrow = 1) +
  labs(
    x     = "Speaker Condition",
    y     = "Match rate",
    title = "Human vs LLM Match Rates by Probing Method",
    subtitle = "Dashed line = chance (1/3). Error bars = bootstrapped 95% CIs."
  ) +
  theme_csp() +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(size = 10, face = "bold"),
    axis.text.x     = element_text(size = 9)
  ) +
  guides(fill = guide_legend(nrow = 1))

ggsave(
  file.path(output_dir, "main_result_plot.png"),
  plot = p_main, width = 16, height = 6, dpi = 300
)
cat("Saved main result plot to:", file.path(output_dir, "main_result_plot.png"), "\n")

# ---------------------------------------------------------------------------
# 10. FOCUSED COMPARISON: direct_label vs meta_label
#     Shows the effect of using role names vs abstract labels
# ---------------------------------------------------------------------------
focus_dl_ml_cot <- llm_match_ci %>%
  filter(source %in% c("direct_label", "meta_label", "chain_of_thought")) %>%
  bind_rows(
    human_ci_for_main %>%
      crossing(model = unique(llm_match_ci$model))
  ) %>%
  mutate(
    condition = factor(condition, levels = c("high", "low", "info")),
    source    = factor(source, levels = c("Human", "direct_label", "meta_label", "chain_of_thought"))
  )

focus_palette_dl <- c(
  "Human"            = project_colors[6],
  "direct_label"     = project_colors[3],
  "meta_label"       = project_colors[10],
  "chain_of_thought" = project_colors[1]
)
focus_labels_dl <- c(
  "Human" = "Human",
  "direct_label" = "Direct label\n(Student/Teacher/Examiner)",
  "meta_label" = "Meta label\n(high/low/info)",
  "chain_of_thought" = "Chain of thought\n(Analysis + Answer)"
)

p_focus_dl <- ggplot(focus_dl_ml_cot, aes(
  x = condition, y = match_rate, fill = source
)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.8),
    width = 0.2, linewidth = 0.3
  ) +
  geom_hline(yintercept = 1 / 3, linetype = "dashed", colour = "grey40",
             linewidth = 0.3) +
  scale_fill_manual(values = focus_palette_dl, labels = focus_labels_dl, name = NULL) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  facet_wrap(~ model, nrow = 1) +
  labs(
    x        = "Speaker Condition",
    y        = "Match rate",
    title    = "Direct Label vs Meta Label vs Chain of Thought",
    subtitle = "Comparing response format (role names vs abstract labels) and reasoning (CoT) effects."
  ) +
  theme_csp() +
  theme(legend.position = "bottom")

ggsave(
  file.path(output_dir, "focus_direct_meta_cot.png"),
  plot = p_focus_dl, width = 14, height = 5, dpi = 300
)
cat("Saved direct_label vs meta_label vs CoT plot.\n")

# ---------------------------------------------------------------------------
# 11. FOCUSED COMPARISON: self_reported_confidence vs log_probs
# ---------------------------------------------------------------------------
focus_sc_lp <- llm_match_ci %>%
  filter(source %in% c("self_reported_confidence", "log_probs")) %>%
  bind_rows(
    human_ci_for_main %>%
      crossing(model = unique(llm_match_ci$model))
  ) %>%
  mutate(
    condition = factor(condition, levels = c("high", "low", "info")),
    source    = factor(source, levels = c("Human", "self_reported_confidence", "log_probs"))
  )

focus_palette_sc <- c(
  "Human"                    = project_colors[6],
  "self_reported_confidence" = project_colors[2],
  "log_probs"                = project_colors[5]
)
focus_labels_sc <- c(
  "Human" = "Human",
  "self_reported_confidence" = "Self-reported\nconfidence",
  "log_probs" = "Log probs"
)

p_focus_sc <- ggplot(focus_sc_lp, aes(
  x = condition, y = match_rate, fill = source
)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.8),
    width = 0.2, linewidth = 0.3
  ) +
  geom_hline(yintercept = 1 / 3, linetype = "dashed", colour = "grey40",
             linewidth = 0.3) +
  scale_fill_manual(values = focus_palette_sc, labels = focus_labels_sc, name = NULL) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  facet_wrap(~ model, nrow = 1) +
  labs(
    x        = "Speaker Condition",
    y        = "Match rate",
    title    = "Self-Reported Confidence vs Log Probs",
    subtitle = "Both methods use the same prompt (direct_label format). Does the output format affect classification?"
  ) +
  theme_csp() +
  theme(legend.position = "bottom")

ggsave(
  file.path(output_dir, "focus_confidence_vs_logprobs.png"),
  plot = p_focus_sc, width = 14, height = 5, dpi = 300
)
cat("Saved self_reported_confidence vs log_probs plot.\n")

# ---------------------------------------------------------------------------
# 12. Heatmap: match rates across the grid
# ---------------------------------------------------------------------------
heatmap_data <- llm_match %>%
  group_by(model, prompt_template, temperature) %>%
  summarise(
    overall_match_rate = weighted.mean(match_rate, n_trials),
    .groups = "drop"
  ) %>%
  mutate(
    grid_label = paste0(prompt_template, "\n(T=", temperature, ")")
  )

p_heatmap <- ggplot(heatmap_data, aes(
  x = factor(temperature),
  y = prompt_template,
  fill = overall_match_rate
)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", overall_match_rate)),
            size = 3.5, color = "black") +
  ggplot2::scale_fill_gradient2(
    low = "#d73027", mid = "#fee08b", high = "#1a9850",
    midpoint = 1 / 3,
    limits = c(0, 1),
    name = "Match Rate"
  ) +
  facet_wrap(~ model) +
  labs(
    x        = "Temperature",
    y        = "Probing Method",
    title    = "LLM Match Rate Heatmap (Probing Method x Temperature)",
    subtitle = "Match rate = proportion where LLM chosen_speaker matches ground-truth condition"
  ) +
  theme_csp() +
  theme(
    axis.text.y = element_text(size = 9),
    panel.grid  = element_blank()
  )

ggsave(
  file.path(output_dir, "grid_heatmap.png"),
  plot = p_heatmap,
  width = 8, height = 5,
  dpi = 300
)
cat("Saved heatmap to:", file.path(output_dir, "grid_heatmap.png"), "\n")

# ---------------------------------------------------------------------------
# 11. Per-condition heatmaps
# ---------------------------------------------------------------------------
p_heatmap_cond <- ggplot(llm_match, aes(
  x = factor(temperature),
  y = prompt_template,
  fill = match_rate
)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", match_rate)),
            size = 3, color = "black") +
  ggplot2::scale_fill_gradient2(
    low = "#d73027", mid = "#fee08b", high = "#1a9850",
    midpoint = 1 / 3,
    limits = c(0, 1),
    name = "Match Rate"
  ) +
  facet_grid(model ~ condition) +
  labs(
    x        = "Temperature",
    y        = "Probing Method",
    title    = "LLM Match Rate by Condition",
    subtitle = "Columns = condition (high / info / low). Color scale midpoint = chance (1/3)."
  ) +
  theme_csp() +
  theme(
    axis.text.y = element_text(size = 8),
    panel.grid    = element_blank(),
    strip.text    = element_text(size = 9)
  )

ggsave(
  file.path(output_dir, "grid_heatmap_by_condition.png"),
  plot = p_heatmap_cond,
  width = 10, height = 6,
  dpi = 300
)

# ---------------------------------------------------------------------------
# 12. Self-reported confidence method: confidence score analysis
# ---------------------------------------------------------------------------
calibration_data <- llm_valid %>%
  filter(prompt_template == "self_reported_confidence") %>%
  filter(!is.na(confidence))

if (nrow(calibration_data) > 0) {
  calibration_data <- calibration_data %>%
    mutate(confidence_num = as.numeric(confidence))

  if (any(!is.na(calibration_data$confidence_num))) {
    p_cal <- ggplot(calibration_data, aes(
      x = condition, y = confidence_num, fill = condition
    )) +
      geom_boxplot(alpha = 0.7, outlier.size = 0.8) +
      scale_fill_manual(values = cb_palette) +
      facet_wrap(~ model, labeller = label_both) +
      labs(
        x     = "Condition",
        y     = "Self-reported Confidence",
        title = "LLM Confidence Scores (Self-Reported Confidence Method)"
      ) +
      theme_csp() +
      theme(
        legend.position = "none",
        plot.title = element_text(face = "bold")
      )

    ggsave(
      file.path(output_dir, "self_reported_confidence.png"),
      plot = p_cal, width = 8, height = 5,
      dpi = 300
    )
    cat("Saved self-reported confidence plot.\n")
  } else {
    cat("Self-reported confidence method found but values are not numeric. Skipping plot.\n")
  }
} else {
  cat("No self-reported confidence data found. Skipping confidence analysis.\n")
}

# ---------------------------------------------------------------------------
# 13. Chain-of-thought: flag reasoning responses
# ---------------------------------------------------------------------------
reasoning_methods <- c("chain_of_thought")
reasoning_data <- llm_raw %>%
  filter(prompt_template %in% reasoning_methods)

if (nrow(reasoning_data) > 0) {
  reasoning_summary <- reasoning_data %>%
    mutate(
      has_reasoning    = !is.na(raw_response) & nchar(raw_response) > 20,
      is_invalid_resp  = is_invalid
    ) %>%
    group_by(model, prompt_template, temperature) %>%
    summarise(
      total           = n(),
      n_with_reasoning = sum(has_reasoning),
      n_invalid       = sum(is_invalid_resp),
      pct_with_reasoning = mean(has_reasoning) * 100,
      pct_invalid     = mean(is_invalid_resp) * 100,
      .groups = "drop"
    )

  cat("\n--- Reasoning method quality ---\n")
  print(reasoning_summary)

  write_csv(reasoning_summary, file.path(output_dir, "reasoning_method_quality.csv"))
}

cat("\n=== Grid comparison analysis complete ===\n")
