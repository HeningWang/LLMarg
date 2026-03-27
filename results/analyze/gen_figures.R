suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(cspplot)
  library(scales)
})

theme_set(theme_csp())
project_colors <- cspplot::list_colors() |> pull(hex)

# High-arg: shimmer dark 2 (#993333, deep red)  — warm
# Low-arg:  opal dark 2   (#66A3A3, teal)       — cool
col_high <- project_colors[14] # shimmer dark 2 = #993333
col_low <- project_colors[8] # opal dark 2    = #66A3A3

# ---- Data loading & prep ----
d_exps <- read.csv("../../data/data_experiment2/cleaned_data_1and2.csv") |>
  mutate(
    arrayCondition = case_when(
      array_size_condition == "wideShort" ~ "5 \u00d7 12",
      array_size_condition == "narrowShort" ~ "5 \u00d7 6",
      array_size_condition == "wideLong" ~ "11 \u00d7 12",
      array_size_condition == "narrowLong" ~ "11 \u00d7 6"
    ),
    condition = ifelse(condition, "High-arg", "Low-arg") |>
      factor(levels = c("High-arg", "Low-arg"))
  )

d_exps$response <- d_exps$response |>
  str_remove_all("\\[|\\]|'") |>
  str_replace_all(",\\s*", " : ") |>
  str_trim()

# ---- Response ordering: by (p_High - p_Low) descending ----
# Pooled across all array conditions for a stable estimate
diff_order <- d_exps |>
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

# ---- Summary stats with binomial SE ----
# Complete all (condition × arrayCondition × response) combinations so that
# missing cells get a tiny stub proportion rather than being absent — otherwise
# position_dodge renders the present bar wider when the partner bar is missing.
eps <- 0.001

sum_stats <- d_exps |>
  group_by(condition, arrayCondition, response) |>
  summarise(n = n(), .groups = "drop") |>
  complete(condition, arrayCondition, response, fill = list(n = 0)) |>
  group_by(condition, arrayCondition) |>
  mutate(
    total      = sum(n),
    proportion = n / total,
    se         = sqrt(proportion * (1 - proportion) / total),
    ci_lo      = ifelse(n == 0, NA_real_, pmax(0, proportion - 1.96 * se)),
    ci_hi      = ifelse(n == 0, NA_real_, pmin(1, proportion + 1.96 * se)),
    # replace true zeros with a tiny stub so the bar slot is always rendered
    proportion = ifelse(n == 0, eps, proportion)
  ) |>
  ungroup() |>
  mutate(response = factor(response, levels = diff_order))

# ---- Shared plotting function ----
make_plot <- function(data_sub) {
  ggplot(data_sub, aes(
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
      fill = "Condition"
    ) +
    # limits = rev: first level of diff_order (most High-dominant) ends up at top after coord_flip
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
      legend.spacing.x = unit(0.2, "cm")
    )
}

# ---- Main paper figure: 5 × 12 only ----
data_5x12 <- sum_stats |> filter(arrayCondition == "5 \u00d7 12")
p_main <- make_plot(data_5x12)

ggsave("../../data/pics/barplot_5x12_main.pdf", p_main, width = 3.5, height = 4)
ggsave("../../data/pics/barplot_5x12_main.png", p_main, width = 3.5, height = 4, dpi = 300)
cat("Saved barplot_5x12_main.pdf/.png\n")

# ---- Appendix figures: one per remaining array ----
appendix_arrays <- c("5 \u00d7 6", "11 \u00d7 12", "11 \u00d7 6")
file_stems <- c("barplot_5x6", "barplot_11x12", "barplot_11x6")

for (i in seq_along(appendix_arrays)) {
  arr <- appendix_arrays[i]
  fstem <- file_stems[i]
  sub <- sum_stats |> filter(arrayCondition == arr)
  p <- make_plot(sub)
  ggsave(paste0("../../data/pics/", fstem, ".pdf"), p, width = 3.5, height = 4)
  ggsave(paste0("../../data/pics/", fstem, ".png"), p, width = 3.5, height = 4, dpi = 300)
  cat("Saved", fstem, "\n")
}
