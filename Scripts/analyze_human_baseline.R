# =============================================================================
# analyze_human_baseline.R
# Processes human listener-side data to create baseline statistics and plots.
# Run from the Scripts/ directory or the project root.
# =============================================================================

library(tidyverse)
library(cspplot)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
project_root <- here::here()
if (!dir.exists(file.path(project_root, "project"))) {
  # Fallback: assume we are inside Scripts/

  project_root <- dirname(getwd())
}

data_path    <- file.path(project_root, "project", "data", "data_listenerside.csv")
output_dir   <- file.path(project_root, "results", "analyze")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Read & clean human data
# ---------------------------------------------------------------------------
if (!file.exists(data_path)) {
  stop("Human data file not found at: ", data_path,
       "\nMake sure the file exists before running this script.")
}

raw <- read_csv(data_path, show_col_types = FALSE)

# Keep only experimental trials (drop training/sample rows)
human <- raw %>%
  filter(!str_starts(condition, "sample\\.")) %>%
  filter(!is.na(response)) %>%
  select(submission_id, condition, response, itemID, Q1, Q2, A, studentsArray)

cat("Experimental trials after filtering:", nrow(human), "\n")
cat("Unique participants:", n_distinct(human$submission_id), "\n")
cat("Unique items:", n_distinct(human$itemID), "\n")

# ---------------------------------------------------------------------------
# 2. Map human response labels to condition labels
# ---------------------------------------------------------------------------
response_to_label <- c(
  "Teacher"  = "high",
  "Student"  = "low",
  "Examiner" = "info"
)

human <- human %>%
  mutate(
    response_label = response_to_label[response],
    is_match       = (response_label == condition)
  )

# ---------------------------------------------------------------------------
# 3. Per-condition response distribution
# ---------------------------------------------------------------------------
condition_response_dist <- human %>%
  count(condition, response_label) %>%
  group_by(condition) %>%
  mutate(
    total = sum(n),
    proportion = n / total
  ) %>%
  ungroup()

cat("\n--- Response distribution by condition ---\n")
print(condition_response_dist)

# ---------------------------------------------------------------------------
# 4. Match rate per condition
# ---------------------------------------------------------------------------
match_rate <- human %>%
  group_by(condition) %>%
  summarise(
    n_trials   = n(),
    n_match    = sum(is_match),
    match_rate = mean(is_match),
    .groups    = "drop"
  )

cat("\n--- Match rates ---\n")
print(match_rate)

# ---------------------------------------------------------------------------
# 5. Per-item response distribution
# ---------------------------------------------------------------------------
item_response_dist <- human %>%
  count(itemID, condition, response_label) %>%
  group_by(itemID, condition) %>%
  mutate(
    total = sum(n),
    proportion = n / total
  ) %>%
  ungroup()

# ---------------------------------------------------------------------------
# 6. Save summary CSV
# ---------------------------------------------------------------------------
summary_out <- condition_response_dist %>%
  left_join(match_rate, by = "condition") %>%
  select(condition, response_label, n, total, proportion, match_rate)

write_csv(summary_out, file.path(output_dir, "human_baseline_summary.csv"))
cat("\nSaved summary to:", file.path(output_dir, "human_baseline_summary.csv"), "\n")

# Also save item-level data for later comparison
write_csv(item_response_dist, file.path(output_dir, "human_item_distributions.csv"))

# ---------------------------------------------------------------------------
# 7. Bootstrap 95% CIs for response proportions
# ---------------------------------------------------------------------------
set.seed(42)
n_boot <- 2000

boot_ci <- human %>%
  group_by(condition, response_label) %>%
  group_modify(function(.data, .key) {
    # Bootstrap: resample participants, recompute proportion
    participants <- unique(human$submission_id)
    boot_props <- replicate(n_boot, {
      sampled_ids <- sample(participants, length(participants), replace = TRUE)
      boot_data <- human %>%
        filter(submission_id %in% sampled_ids) %>%
        filter(condition == .key$condition)
      mean(boot_data$response_label == .key$response_label, na.rm = TRUE)
    })
    tibble(
      ci_lower = quantile(boot_props, 0.025, na.rm = TRUE),
      ci_upper = quantile(boot_props, 0.975, na.rm = TRUE)
    )
  }) %>%
  ungroup()

plot_data <- condition_response_dist %>%
  left_join(boot_ci, by = c("condition", "response_label"))

# ---------------------------------------------------------------------------
# 8. Publication-quality bar plot (cspplot theme)
# ---------------------------------------------------------------------------
theme_set(theme_csp())
project_colors <- cspplot::list_colors() |> pull(hex)

csp_palette <- c(
  "high" = project_colors[3],   # crayola
  "low"  = project_colors[1],   # glaucous
  "info" = project_colors[4]    # fern
)

# Order factors for display
plot_data <- plot_data %>%
  mutate(
    condition      = factor(condition, levels = c("high", "low", "info")),
    response_label = factor(response_label, levels = c("high", "low", "info"))
  )

p <- ggplot(plot_data, aes(x = condition, y = proportion, fill = response_label)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.8),
    width = 0.25,
    linewidth = 0.4
  ) +
  geom_hline(yintercept = 1 / 3, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  scale_fill_manual(
    values = csp_palette,
    labels = c("high" = "Teacher (high)", "low" = "Student (low)", "info" = "Examiner (info)"),
    name   = "Response"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = "Speaker Condition",
    y = "Proportion of responses",
    title = "Human Listener Responses by Speaker Condition",
    subtitle = "Dashed line = chance level (1/3). Error bars = bootstrapped 95% CIs."
  ) +
  theme_csp() +
  theme(
    legend.position = "bottom"
  )

ggsave(
  file.path(output_dir, "human_baseline_plot.png"),
  plot = p, width = 7, height = 5, dpi = 300
)
cat("Saved plot to:", file.path(output_dir, "human_baseline_plot.png"), "\n")

cat("\n=== Human baseline analysis complete ===\n")
