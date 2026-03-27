suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(cspplot)
  library(scales)
  library(patchwork)
})

theme_set(theme_csp())
project_colors <- cspplot::list_colors() |> pull(hex)

# Colors matching CogSci paper (gen_figures.R)
col_high <- project_colors[14] # shimmer dark 2 = #993333
col_low  <- project_colors[8]  # opal dark 2    = #66A3A3

project_root <- here::here()
if (!dir.exists(file.path(project_root, "project"))) {
  project_root <- dirname(getwd())
}
output_dir <- file.path(project_root, "results", "analyze")

# =========================================================================
# 1. HUMAN DATA (from cleaned_data_1and2.csv, wideShort = 5x12 only)
# =========================================================================
human_path <- "/Users/heningwang/Documents/GitHub/argumentative_language/data/data_experiment2/cleaned_data_1and2.csv"
d_human <- read.csv(human_path) |>
  filter(array_size_condition == "wideShort") |>
  mutate(
    condition = ifelse(condition == 1, "High-arg", "Low-arg") |>
      factor(levels = c("High-arg", "Low-arg")),
    response = response |>
      str_remove_all("\\[|\\]|'") |>
      str_replace_all(",\\s*", " : ") |>
      str_trim()
  )

cat("Human speaker data (wideShort/5x12):", nrow(d_human), "trials\n")

# =========================================================================
# 2. LLM DATA
# =========================================================================
llm_files <- list.files(
  file.path(project_root, "results"),
  pattern = "^speaker_.*\\.csv$",
  full.names = TRUE
)
llm_files <- llm_files[!str_detect(llm_files, "013155")]

llm_raw <- purrr::map_dfr(llm_files, function(path) {
  df <- readr::read_csv(path, show_col_types = FALSE)
  model_name <- str_extract(basename(path), "speaker_(.+)_\\d{8}", group = 1)
  df |> mutate(model = model_name)
})

llm_data <- llm_raw |>
  filter(is_valid == TRUE | is_valid == "True") |>
  mutate(
    oq  = str_to_lower(oq),
    iq  = str_to_lower(iq),
    adj = str_to_lower(adj),
    condition = ifelse(framing == "high", "High-arg", "Low-arg") |>
      factor(levels = c("High-arg", "Low-arg")),
    response = paste(oq, iq, adj, sep = " : ")
  )

cat("LLM valid trials:", nrow(llm_data), "\n\n")

# =========================================================================
# 3. SHARED RESPONSE ORDERING (by High-Low difference, from human data)
# =========================================================================
diff_order <- d_human |>
  group_by(condition, response) |>
  summarise(n = n(), .groups = "drop") |>
  group_by(condition) |>
  mutate(prop = n / sum(n)) |>
  ungroup() |>
  select(condition, response, prop) |>
  pivot_wider(names_from = condition, values_from = prop, values_fill = 0) |>
  mutate(diff = `High-arg` - `Low-arg`) |>
  arrange(desc(diff)) |>
  pull(response)

# =========================================================================
# 4. PLOTTING FUNCTION (matching gen_figures.R style)
# =========================================================================
eps <- 0.001

make_speaker_plot <- function(data, title_text = "") {
  sum_stats <- data |>
    group_by(condition, response) |>
    summarise(n = n(), .groups = "drop") |>
    complete(condition, response = diff_order, fill = list(n = 0)) |>
    group_by(condition) |>
    mutate(
      total      = sum(n),
      proportion = n / total,
      se         = sqrt(proportion * (1 - proportion) / total),
      ci_lo      = ifelse(n == 0, NA_real_, pmax(0, proportion - 1.96 * se)),
      ci_hi      = ifelse(n == 0, NA_real_, pmin(1, proportion + 1.96 * se)),
      proportion = ifelse(n == 0, eps, proportion)
    ) |>
    ungroup() |>
    mutate(response = factor(response, levels = diff_order))

  ggplot(sum_stats, aes(
    x = response, y = proportion,
    fill = condition, group = condition
  )) +
    geom_col(position = position_dodge(width = 0.85), width = 0.8) +
    geom_errorbar(
      aes(ymin = ci_lo, ymax = ci_hi),
      position = position_dodge(width = 0.85),
      width = 0.25, linewidth = 0.35, colour = "grey20"
    ) +
    scale_fill_manual(
      values = c("High-arg" = col_high, "Low-arg" = col_low)
    ) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, NA),
      expand = expansion(mult = c(0, 0.05))
    ) +
    labs(
      x    = NULL,
      y    = "Proportion of responses",
      fill = "Condition",
      title = title_text
    ) +
    scale_x_discrete(limits = rev) +
    coord_flip() +
    theme_csp() +
    theme(
      axis.text.y      = element_text(size = 7),
      axis.text.x      = element_text(size = 7),
      axis.title.x     = element_text(size = 7),
      axis.ticks.y     = element_blank(),
      legend.position  = "top",
      legend.key.width  = unit(0.4, "cm"),
      legend.key.height = unit(0.2, "cm"),
      legend.text      = element_text(size = 7),
      legend.title     = element_text(size = 7),
      legend.spacing.x = unit(0.2, "cm"),
      plot.title       = element_text(size = 9, face = "bold")
    )
}

# =========================================================================
# 5. HUMAN BASELINE PLOT (5x12, matching CogSci paper style)
# =========================================================================
p_human <- make_speaker_plot(d_human, "Human (5 x 12)")

ggsave(file.path(output_dir, "speaker_human_baseline.png"),
       p_human, width = 4, height = 5, dpi = 300)
cat("Saved speaker human baseline plot.\n")

# =========================================================================
# 6. MAIN COMPARISON: Human + 4 LLMs side by side
# =========================================================================
models <- sort(unique(llm_data$model))

# Create individual plots
p_human_panel <- make_speaker_plot(d_human, "Human")

llm_panels <- purrr::map(models, function(m) {
  make_speaker_plot(llm_data |> filter(model == m), m)
})

# Combine: Human + 4 models
all_panels <- c(list(p_human_panel), llm_panels)
p_combined <- wrap_plots(all_panels, nrow = 1) +
  plot_annotation(
    title = "Speaker Production: Utterance Choice Distribution (5 x 12)",
    subtitle = "Bars show high-arg (red) and low-arg (teal) framing. Ordered by human High-Low difference.",
    theme = theme(
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 9, colour = "grey40")
    )
  )

ggsave(file.path(output_dir, "speaker_main_comparison.png"),
       p_combined, width = 18, height = 6, dpi = 300)
cat("Saved speaker main comparison plot.\n")

# =========================================================================
# 7. OUTPUT FORMAT COMPARISON: full_sentence vs three_blanks per model
# =========================================================================
format_panels <- list()

for (m in models) {
  for (fmt in c("full_sentence", "three_blanks")) {
    sub <- llm_data |> filter(model == m, output_format == fmt)
    if (nrow(sub) > 0) {
      label <- paste0(m, "\n(", fmt, ")")
      format_panels[[length(format_panels) + 1]] <- make_speaker_plot(sub, label)
    }
  }
}

p_format <- wrap_plots(format_panels, nrow = 2) +
  plot_annotation(
    title = "Output Format Comparison: Full Sentence vs Three Blanks",
    subtitle = "Does output format affect utterance choice distribution?",
    theme = theme(
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 9, colour = "grey40")
    )
  )

ggsave(file.path(output_dir, "speaker_format_comparison.png"),
       p_format, width = 18, height = 10, dpi = 300)
cat("Saved speaker format comparison plot.\n")

cat("\n=== Speaker analysis v2 complete ===\n")
