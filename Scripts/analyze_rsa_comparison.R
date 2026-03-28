# =============================================================================
# analyze_rsa_comparison.R
# Compares RSA model predictions vs Human data and vs LLM data
# for both listener and speaker experiments.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(cspplot)
  library(scales)
})

theme_set(theme_csp())
project_colors <- cspplot::list_colors() |> pull(hex)

project_root <- here::here()
if (!dir.exists(file.path(project_root, "project"))) {
  project_root <- dirname(getwd())
}

results_dir <- file.path(project_root, "results")
output_dir  <- file.path(project_root, "results", "analyze")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ── Utility: JSD ─────────────────────────────────────────────────────────────
js_divergence <- function(p, q) {
  eps <- 1e-10
  p <- pmax(p, eps); q <- pmax(q, eps)
  p <- p / sum(p);   q <- q / sum(q)
  m <- (p + q) / 2
  (sum(p * log2(p / m)) + sum(q * log2(q / m))) / 2
}

# ==========================================================================
# PART 1: LISTENER EXPERIMENT
# ==========================================================================
cat("\n========== LISTENER EXPERIMENT ==========\n")

# ── 1a. Load RSA predictions ──────────────────────────────────────────────
rsa_listener <- read_csv(
  file.path(output_dir, "rsa_predictions_per_item.csv"),
  show_col_types = FALSE
)

# ── 1b. Load human item-level data ───────────────────────────────────────
human_item <- read_csv(
  file.path(output_dir, "human_item_distributions.csv"),
  show_col_types = FALSE
)

# Pivot human to wide: one row per item with p_high, p_info, p_low
human_wide <- human_item |>
  select(itemID, condition, response_label, proportion) |>
  pivot_wider(
    names_from  = response_label,
    values_from = proportion,
    names_prefix = "h_"
  )

# ── 1c. Load best LLM: Qwen2.5-7B × meta_label ─────────────────────────
grid_files <- Sys.glob(file.path(results_dir, "grid_*.csv"))

llm_raw <- map_dfr(grid_files, function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  model_name <- basename(path) |>
    str_remove("^grid_") |>
    str_remove("_\\d{8}_\\d{6}\\.csv$")
  df |> mutate(model = model_name)
})

# Best listener config
best_llm_listener <- llm_raw |>
  filter(
    model == "qwen2_5_7b",
    prompt_template == "meta_label",
    !str_starts(chosen_speaker, "Invalid"),
    !str_starts(chosen_speaker, "ERROR"),
    !is.na(chosen_speaker)
  )

# Compute item-level distributions for best LLM
llm_item_dist <- best_llm_listener |>
  count(itemID, condition, chosen_speaker) |>
  group_by(itemID, condition) |>
  mutate(proportion = n / sum(n)) |>
  ungroup() |>
  select(itemID, condition, response_label = chosen_speaker, proportion)

# Ensure all 3 labels present per item
llm_item_wide <- llm_item_dist |>
  complete(
    nesting(itemID, condition),
    response_label = c("high", "info", "low"),
    fill = list(proportion = 0)
  ) |>
  pivot_wider(
    names_from  = response_label,
    values_from = proportion,
    names_prefix = "llm_"
  )

# ── 1d. Compute JSD: RSA vs Human, RSA vs LLM, LLM vs Human ────────────
rsa_models <- unique(rsa_listener$model)

listener_jsd_results <- map_dfr(rsa_models, function(rsa_mod) {
  rsa_sub <- rsa_listener |>
    filter(model == rsa_mod) |>
    select(itemID, condition, p_low, p_info, p_high)

  # Merge with human and LLM
  merged <- rsa_sub |>
    inner_join(human_wide, by = c("itemID", "condition")) |>
    inner_join(llm_item_wide, by = c("itemID", "condition"))

  merged |>
    rowwise() |>
    mutate(
      jsd_rsa_human = js_divergence(
        c(p_high, p_info, p_low),
        c(h_high, h_info, h_low)
      ),
      jsd_rsa_llm = js_divergence(
        c(p_high, p_info, p_low),
        c(llm_high, llm_info, llm_low)
      ),
      jsd_llm_human = js_divergence(
        c(llm_high, llm_info, llm_low),
        c(h_high, h_info, h_low)
      )
    ) |>
    ungroup() |>
    mutate(rsa_model = rsa_mod)
})

# Summary per RSA model
listener_summary <- listener_jsd_results |>
  group_by(rsa_model) |>
  summarise(
    mean_jsd_rsa_human = mean(jsd_rsa_human),
    mean_jsd_rsa_llm   = mean(jsd_rsa_llm),
    mean_jsd_llm_human = mean(jsd_llm_human),
    .groups = "drop"
  ) |>
  arrange(mean_jsd_rsa_human)

cat("\n--- Listener: Mean JSD per RSA model ---\n")
cat("(LLM = Qwen2.5-7B × meta_label)\n\n")
print(listener_summary, n = Inf)

# Use nonparametric (model-free) as the best RSA model per domain knowledge
best_rsa_listener <- "nonparametric"
cat("\nBest RSA model (listener):", best_rsa_listener, "(model-free)\n")

write_csv(listener_summary, file.path(output_dir, "listener_rsa_comparison_summary.csv"))
write_csv(listener_jsd_results, file.path(output_dir, "listener_rsa_comparison_detail.csv"))

# ── 1e. Item-level scatterplots ──────────────────────────────────────────

# Focus on best RSA model
best_rsa_data <- listener_jsd_results |>
  filter(rsa_model == best_rsa_listener)

# Plot 1: RSA vs Human (correct response proportion)
# For each item, the "correct" response matches its condition
rsa_correct <- best_rsa_data |>
  mutate(
    rsa_correct = case_when(
      condition == "high" ~ p_high,
      condition == "low"  ~ p_low,
      condition == "info" ~ p_info
    ),
    human_correct = case_when(
      condition == "high" ~ h_high,
      condition == "low"  ~ h_low,
      condition == "info" ~ h_info
    ),
    llm_correct = case_when(
      condition == "high" ~ llm_high,
      condition == "low"  ~ llm_low,
      condition == "info" ~ llm_info
    )
  )

cond_colors <- c(
  "high" = project_colors[3],
  "low"  = project_colors[1],
  "info" = project_colors[4]
)

p_rsa_human <- ggplot(rsa_correct, aes(
  x = human_correct, y = rsa_correct, colour = condition
)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_colour_manual(values = cond_colors) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    x     = "Human P(correct response)",
    y     = paste0("RSA (", best_rsa_listener, ") P(correct response)"),
    title = paste0("RSA (", best_rsa_listener, ") vs Human"),
    subtitle = sprintf("r = %.3f", cor(rsa_correct$rsa_correct, rsa_correct$human_correct)),
    colour = "Condition"
  ) +
  theme_csp()

p_rsa_llm <- ggplot(rsa_correct, aes(
  x = llm_correct, y = rsa_correct, colour = condition
)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_colour_manual(values = cond_colors) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    x     = "LLM (Qwen × meta_label) P(correct response)",
    y     = paste0("RSA (", best_rsa_listener, ") P(correct response)"),
    title = paste0("RSA (", best_rsa_listener, ") vs LLM"),
    subtitle = sprintf("r = %.3f", cor(rsa_correct$rsa_correct, rsa_correct$llm_correct)),
    colour = "Condition"
  ) +
  theme_csp()

p_llm_human <- ggplot(rsa_correct, aes(
  x = human_correct, y = llm_correct, colour = condition
)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_colour_manual(values = cond_colors) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    x     = "Human P(correct response)",
    y     = "LLM (Qwen × meta_label) P(correct response)",
    title = "LLM vs Human",
    subtitle = sprintf("r = %.3f", cor(rsa_correct$llm_correct, rsa_correct$human_correct)),
    colour = "Condition"
  ) +
  theme_csp()

p_listener_scatter <- p_rsa_human + p_rsa_llm + p_llm_human +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Listener: Item-Level Correct-Response Probability",
    subtitle = "30 items × 3 conditions. Dashed = identity line.",
    theme = theme(
      plot.title    = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 10, colour = "grey40")
    )
  ) &
  theme(legend.position = "bottom")

ggsave(
  file.path(output_dir, "listener_rsa_scatter.png"),
  p_listener_scatter, width = 15, height = 6, dpi = 300
)
cat("Saved listener scatter plot.\n")

# ── 1f. Full distribution comparison (all 3 probabilities) ──────────────
# For the best RSA model, plot predicted vs observed distribution per item
# using JSD as the distance measure

jsd_by_condition <- listener_jsd_results |>
  filter(rsa_model == best_rsa_listener) |>
  group_by(condition) |>
  summarise(
    mean_jsd_rsa_human = mean(jsd_rsa_human),
    mean_jsd_rsa_llm   = mean(jsd_rsa_llm),
    mean_jsd_llm_human = mean(jsd_llm_human),
    .groups = "drop"
  )

cat("\n--- JSD by condition (best RSA:", best_rsa_listener, ") ---\n")
print(jsd_by_condition)

# ── 1g. Bar chart: mean JSD across all RSA models ───────────────────────
listener_long <- listener_summary |>
  pivot_longer(
    cols = starts_with("mean_jsd"),
    names_to = "comparison",
    values_to = "mean_jsd"
  ) |>
  mutate(
    comparison = recode(comparison,
      "mean_jsd_rsa_human" = "RSA vs Human",
      "mean_jsd_rsa_llm"   = "RSA vs LLM",
      "mean_jsd_llm_human" = "LLM vs Human"
    )
  )

comp_colors <- c(
  "RSA vs Human" = project_colors[3],
  "RSA vs LLM"   = project_colors[1],
  "LLM vs Human" = project_colors[5]
)

p_listener_bar <- ggplot(listener_long, aes(
  x = reorder(rsa_model, mean_jsd), y = mean_jsd, fill = comparison
)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values = comp_colors) +
  coord_flip() +
  labs(
    x     = "RSA Model",
    y     = "Mean Jensen-Shannon Divergence",
    title = "Listener: Model Comparison (Mean JSD)",
    subtitle = "LLM = Qwen2.5-7B × meta_label. Lower = closer.",
    fill  = NULL
  ) +
  theme_csp() +
  theme(legend.position = "bottom")

ggsave(
  file.path(output_dir, "listener_rsa_bar.png"),
  p_listener_bar, width = 8, height = 5, dpi = 300
)
cat("Saved listener bar chart.\n")

# ==========================================================================
# PART 2: SPEAKER EXPERIMENT
# ==========================================================================
cat("\n\n========== SPEAKER EXPERIMENT ==========\n")

# ── 2a. Load human speaker data (wideShort/5x12 only) ───────────────────
human_speaker_path <- "/Users/heningwang/Documents/GitHub/argumentative_language/data/data_experiment2/cleaned_data_1and2.csv"
d_human_full <- read.csv(human_speaker_path)

d_human_ws <- d_human_full |>
  filter(array_size_condition == "wideShort") |>
  mutate(
    condition = ifelse(condition == 1, "High-arg", "Low-arg"),
    response = response |>
      str_remove_all("\\[|\\]|'") |>
      str_replace_all(",\\s*", " : ") |>
      str_trim()
  )

cat("Human speaker (wideShort):", nrow(d_human_ws), "trials\n")

# ── 2b. Load RSA speaker predictions (PPS files) ────────────────────────
pps_dir <- "/Users/heningwang/Documents/GitHub/argumentative_language/analysis/posterior_predictive_samples"

rsa_speaker_models <- c(
  "base"          = "pps_base_hierarchical.csv",
  "lr"            = "pps_lr_argstrength_hierarchical.csv",
  "prag"          = "pps_prag_argstrength_hierarchical.csv",
  "maximin"       = "pps_maximin_argstrength_hierarchical.csv",
  "nonparametric" = "pps_nonparametric_argstrength_hierarchical.csv"
)

# The PPS rows align with cleaned_data_1and2.csv rows
# Filter to wideShort indices
ws_mask <- d_human_full$array_size_condition == "wideShort"
ws_indices <- which(ws_mask)  # 1-based R indices
ws_conditions <- ifelse(d_human_full$condition[ws_mask] == 1, "High-arg", "Low-arg")

# For each RSA model, extract wideShort rows and compute
# mean predicted distribution per condition
rsa_speaker_dists <- map_dfr(names(rsa_speaker_models), function(mod_name) {
  pps <- read_csv(
    file.path(pps_dir, rsa_speaker_models[mod_name]),
    show_col_types = FALSE
  )
  # Drop the row-index column (first unnamed column)
  if (names(pps)[1] == "" || names(pps)[1] == "X1" || str_detect(names(pps)[1], "^\\.\\.\\.")) {
    pps <- pps[, -1]
  }

  # Extract wideShort rows
  pps_ws <- pps[ws_indices, ]
  pps_ws$condition <- ws_conditions

  # Average predicted distribution per condition
  pps_ws |>
    group_by(condition) |>
    summarise(across(everything(), mean), .groups = "drop") |>
    pivot_longer(-condition, names_to = "utterance", values_to = "rsa_prob") |>
    mutate(rsa_model = mod_name)
})

# ── 2c. Compute human speaker distribution per condition ─────────────────
# Map human responses to the same utterance format as PPS columns
# PPS format: "Q1|Q2|right/wrong" → human format: "Q1 : Q2 : right/wrong"
# Need to convert between formats

human_speaker_dist <- d_human_ws |>
  mutate(
    # Convert "most : most : wrong" → "most|most|wrong"
    utterance = str_replace_all(response, " : ", "|")
  ) |>
  count(condition, utterance) |>
  group_by(condition) |>
  mutate(human_prob = n / sum(n)) |>
  ungroup() |>
  select(condition, utterance, human_prob)

# ── 2d. Load best LLM speaker: Gemma3-12B ───────────────────────────────
speaker_files <- Sys.glob(file.path(results_dir, "speaker_*.csv"))
# Exclude the duplicate qwen file (013155)
speaker_files <- speaker_files[!str_detect(speaker_files, "013155")]

llm_speaker_raw <- map_dfr(speaker_files, function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  model_name <- str_extract(basename(path), "speaker_(.+)_\\d{8}", group = 1)
  df |> mutate(model = model_name)
})

best_llm_speaker <- llm_speaker_raw |>
  filter(
    model == "gemma3_12b",
    is_valid == TRUE | is_valid == "True"
  ) |>
  mutate(
    oq  = str_to_lower(oq),
    iq  = str_to_lower(iq),
    adj = str_to_lower(adj),
    condition = ifelse(framing == "high", "High-arg", "Low-arg"),
    utterance = paste(oq, iq, adj, sep = "|")
  )

llm_speaker_dist <- best_llm_speaker |>
  count(condition, utterance) |>
  group_by(condition) |>
  mutate(llm_prob = n / sum(n)) |>
  ungroup() |>
  select(condition, utterance, llm_prob)

# ── 2e. Compute JSD for speaker: RSA vs Human, RSA vs LLM ───────────────
# Get the full utterance universe
all_utterances <- sort(unique(c(
  rsa_speaker_dists$utterance,
  human_speaker_dist$utterance,
  llm_speaker_dist$utterance
)))

# Fill missing utterances with 0
fill_dist <- function(df, prob_col, conditions = c("High-arg", "Low-arg")) {
  crossing(condition = conditions, utterance = all_utterances) |>
    left_join(df, by = c("condition", "utterance")) |>
    mutate(across(all_of(prob_col), ~ replace_na(.x, 0)))
}

human_filled <- fill_dist(human_speaker_dist, "human_prob")
llm_filled   <- fill_dist(llm_speaker_dist, "llm_prob")

speaker_jsd_results <- map_dfr(unique(rsa_speaker_dists$rsa_model), function(mod) {
  rsa_filled <- rsa_speaker_dists |>
    filter(rsa_model == mod) |>
    select(condition, utterance, rsa_prob)
  rsa_filled <- fill_dist(rsa_filled, "rsa_prob")

  map_dfr(c("High-arg", "Low-arg"), function(cond) {
    rsa_p   <- rsa_filled   |> filter(condition == cond) |> arrange(utterance) |> pull(rsa_prob)
    human_p <- human_filled  |> filter(condition == cond) |> arrange(utterance) |> pull(human_prob)
    llm_p   <- llm_filled    |> filter(condition == cond) |> arrange(utterance) |> pull(llm_prob)

    tibble(
      rsa_model      = mod,
      condition      = cond,
      jsd_rsa_human  = js_divergence(rsa_p, human_p),
      jsd_rsa_llm    = js_divergence(rsa_p, llm_p),
      jsd_llm_human  = js_divergence(llm_p, human_p)
    )
  })
})

speaker_summary <- speaker_jsd_results |>
  group_by(rsa_model) |>
  summarise(
    mean_jsd_rsa_human = mean(jsd_rsa_human),
    mean_jsd_rsa_llm   = mean(jsd_rsa_llm),
    mean_jsd_llm_human = mean(jsd_llm_human),
    .groups = "drop"
  ) |>
  arrange(mean_jsd_rsa_human)

cat("\n--- Speaker: Mean JSD per RSA model ---\n")
cat("(LLM = Gemma3-12B)\n\n")
print(speaker_summary, n = Inf)

# Use nonparametric (model-free) as the best RSA model per domain knowledge
best_rsa_speaker <- "nonparametric"
cat("\nBest RSA model (speaker):", best_rsa_speaker, "(model-free)\n")

write_csv(speaker_summary, file.path(output_dir, "speaker_rsa_comparison_summary.csv"))
write_csv(speaker_jsd_results, file.path(output_dir, "speaker_rsa_comparison_detail.csv"))

# ── 2f. Speaker bar chart ────────────────────────────────────────────────
speaker_long <- speaker_summary |>
  pivot_longer(
    cols = starts_with("mean_jsd"),
    names_to = "comparison",
    values_to = "mean_jsd"
  ) |>
  mutate(
    comparison = recode(comparison,
      "mean_jsd_rsa_human" = "RSA vs Human",
      "mean_jsd_rsa_llm"   = "RSA vs LLM",
      "mean_jsd_llm_human" = "LLM vs Human"
    )
  )

p_speaker_bar <- ggplot(speaker_long, aes(
  x = reorder(rsa_model, mean_jsd), y = mean_jsd, fill = comparison
)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values = comp_colors) +
  coord_flip() +
  labs(
    x     = "RSA Model",
    y     = "Mean Jensen-Shannon Divergence",
    title = "Speaker: Model Comparison (Mean JSD)",
    subtitle = "LLM = Gemma3-12B. Lower = closer.",
    fill  = NULL
  ) +
  theme_csp() +
  theme(legend.position = "bottom")

ggsave(
  file.path(output_dir, "speaker_rsa_bar.png"),
  p_speaker_bar, width = 8, height = 5, dpi = 300
)
cat("Saved speaker bar chart.\n")

# ── 2g. Speaker: top utterances comparison (best RSA vs Human vs LLM) ───
# Show top 15 utterances by human frequency, compare distributions

top_utts <- human_speaker_dist |>
  group_by(utterance) |>
  summarise(total_prob = sum(human_prob), .groups = "drop") |>
  slice_max(total_prob, n = 15) |>
  pull(utterance)

rsa_best_speaker <- rsa_speaker_dists |>
  filter(rsa_model == best_rsa_speaker) |>
  select(condition, utterance, rsa_prob)

speaker_top_data <- crossing(
  condition = c("High-arg", "Low-arg"),
  utterance = top_utts
) |>
  left_join(human_speaker_dist, by = c("condition", "utterance")) |>
  left_join(llm_speaker_dist, by = c("condition", "utterance")) |>
  left_join(rsa_best_speaker, by = c("condition", "utterance")) |>
  mutate(across(c(human_prob, llm_prob, rsa_prob), ~ replace_na(.x, 0))) |>
  pivot_longer(
    cols = c(human_prob, llm_prob, rsa_prob),
    names_to = "source",
    values_to = "prob"
  ) |>
  mutate(
    source = recode(source,
      "human_prob" = "Human",
      "llm_prob"   = "LLM (Gemma3)",
      "rsa_prob"   = paste0("RSA (", best_rsa_speaker, ")")
    ),
    utterance_label = str_replace_all(utterance, "\\|", " : "),
    condition = factor(condition, levels = c("High-arg", "Low-arg"))
  )

# Order utterances by High-Low difference (human)
utt_order <- speaker_top_data |>
  filter(source == "Human") |>
  select(utterance_label, condition, prob) |>
  pivot_wider(names_from = condition, values_from = prob, values_fill = 0) |>
  mutate(diff = `High-arg` - `Low-arg`) |>
  arrange(desc(diff)) |>
  pull(utterance_label)

speaker_top_data <- speaker_top_data |>
  mutate(utterance_label = factor(utterance_label, levels = rev(utt_order)))

source_colors <- c(
  "Human"       = project_colors[6],
  "LLM (Gemma3)" = project_colors[3]
)
source_colors[paste0("RSA (", best_rsa_speaker, ")")] <- project_colors[1]

p_speaker_top <- ggplot(speaker_top_data, aes(
  x = utterance_label, y = prob, fill = source
)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ condition, ncol = 2) +
  scale_fill_manual(values = source_colors) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    x     = NULL,
    y     = "Proportion",
    fill  = NULL,
    title = "Speaker: Top 15 Utterances by Source",
    subtitle = "Comparing Human, best RSA, and best LLM distributions"
  ) +
  theme_csp() +
  theme(
    legend.position = "bottom",
    axis.text.y     = element_text(size = 7)
  )

ggsave(
  file.path(output_dir, "speaker_rsa_top_utterances.png"),
  p_speaker_top, width = 12, height = 7, dpi = 300
)
cat("Saved speaker top utterances comparison.\n")

# ==========================================================================
# PART 3: COMBINED SUMMARY TABLE
# ==========================================================================
cat("\n\n========== COMBINED SUMMARY ==========\n")

combined <- bind_rows(
  listener_summary |>
    mutate(experiment = "Listener", llm = "Qwen2.5-7B × meta_label") |>
    select(experiment, rsa_model, llm, everything()),
  speaker_summary |>
    mutate(experiment = "Speaker", llm = "Gemma3-12B") |>
    select(experiment, rsa_model, llm, everything())
)

cat("\n--- Full comparison table ---\n")
print(combined, n = Inf, width = 120)

write_csv(combined, file.path(output_dir, "rsa_llm_comparison_combined.csv"))

cat("\n=== RSA comparison analysis complete ===\n")
cat("Best RSA (listener):", best_rsa_listener, "(model-free)\n")
cat("Best RSA (speaker):", best_rsa_speaker, "(model-free)\n")
cat("Best LLM (listener): Qwen2.5-7B × meta_label\n")
cat("Best LLM (speaker): Gemma3-12B\n")
