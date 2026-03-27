# =============================================================================
# analyze_speaker.R
# Compares LLM speaker-side production to human speaker data.
# Run from the Scripts/ directory.
# =============================================================================

library(tidyverse)
library(cspplot)
library(scales)
library(patchwork)

theme_set(theme_csp())
project_colors <- cspplot::list_colors() |> pull(hex)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
project_root <- here::here()
if (!dir.exists(file.path(project_root, "project"))) {
  project_root <- dirname(getwd())
}

output_dir <- file.path(project_root, "results", "analyze")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Load human speaker data
# ---------------------------------------------------------------------------
human_path <- file.path(project_root, "project", "data", "data_speakerside.csv")
human_raw <- read_csv(human_path, show_col_types = FALSE)

human <- human_raw %>%
  mutate(
    framing = if_else(condition == 1, "high", "low"),
    oq  = str_to_lower(str_split_i(responses, "\\|", 1)),
    iq  = str_to_lower(str_split_i(responses, "\\|", 2)),
    adj = str_to_lower(str_split_i(responses, "\\|", 3)),
    n_students = str_count(studentsArray, "\\|") + 1,
    source = "Human"
  )

cat("Human speaker data:", nrow(human), "trials,",
    n_distinct(human$submission_id), "participants\n")

# ---------------------------------------------------------------------------
# 2. Load LLM speaker data
# ---------------------------------------------------------------------------
llm_files <- list.files(
  file.path(project_root, "results"),
  pattern = "^speaker_.*\\.csv$",
  full.names = TRUE
)

# Exclude the old pilot file (013155)
llm_files <- llm_files[!str_detect(llm_files, "013155")]

if (length(llm_files) == 0) {
  stop("No speaker result files found.")
}

cat("Found", length(llm_files), "speaker result file(s)\n")

llm_raw <- map_dfr(llm_files, function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  model_name <- str_extract(basename(path), "speaker_(.+)_\\d{8}", group = 1)
  df %>% mutate(model = model_name)
})

llm <- llm_raw %>%
  filter(is_valid == TRUE | is_valid == "True") %>%
  mutate(
    oq     = str_to_lower(oq),
    iq     = str_to_lower(iq),
    adj    = str_to_lower(adj),
    source = model
  )

cat("LLM valid trials:", nrow(llm), "\n\n")

# ---------------------------------------------------------------------------
# 3. Compute full utterance distributions (OQ:IQ:ADJ)
# ---------------------------------------------------------------------------
main_palette <- c(
  "Human"       = project_colors[6],   # independence
  "gemma3_12b"  = project_colors[3],   # crayola
  "llama3_1_8b" = project_colors[1],   # glaucous
  "mistral_7b"  = project_colors[2],   # shimmer
  "qwen2_5_7b"  = project_colors[5]    # opal
)

# Human full utterance distribution
human_utt <- human %>%
  mutate(utterance = paste0(oq, ":", iq, ":", adj)) %>%
  count(framing, utterance) %>%
  group_by(framing) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup() %>%
  mutate(model = "Human")

# LLM full utterance distribution (pooled across output formats)
llm_utt <- llm %>%
  mutate(utterance = paste0(oq, ":", iq, ":", adj)) %>%
  count(model, framing, utterance) %>%
  group_by(model, framing) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

# ---------------------------------------------------------------------------
# 4. Main result plot: Full utterance distribution, Human vs LLM
#    x-axis: utterance (OQ:IQ:ADJ), sorted by human frequency
#    Bars: Human + 4 models
#    Faceted by framing (high/low)
# ---------------------------------------------------------------------------

# Get top utterances by human frequency (keep those with >2% in either framing)
top_utts <- human_utt %>%
  group_by(utterance) %>%
  summarise(max_prop = max(proportion), .groups = "drop") %>%
  filter(max_prop > 0.02) %>%
  arrange(desc(max_prop)) %>%
  pull(utterance)

utt_combined <- bind_rows(
  human_utt %>% select(model, framing, utterance, proportion),
  llm_utt %>% select(model, framing, utterance, proportion)
) %>%
  filter(utterance %in% top_utts) %>%
  mutate(
    utterance = factor(utterance, levels = rev(top_utts)),
    framing   = factor(framing, levels = c("high", "low")),
    model     = factor(model, levels = c("Human", sort(unique(llm_utt$model))))
  )

# Fill missing combinations with 0
utt_complete <- utt_combined %>%
  complete(model, framing, utterance, fill = list(proportion = 0))

p_utt <- ggplot(utt_complete, aes(x = utterance, y = proportion, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ framing, nrow = 2, scales = "free_x",
             labeller = labeller(framing = c("high" = "High framing",
                                             "low" = "Low framing"))) +
  scale_fill_manual(values = main_palette, name = "Source") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.05))) +
  coord_flip() +
  labs(
    x     = "Utterance (OQ : IQ : ADJ)",
    y     = "Proportion",
    title = "Speaker Production: Utterance Choice Distribution",
    subtitle = "Human vs LLM. Showing utterances with >2% human usage."
  ) +
  theme_csp() +
  theme(legend.position = "bottom",
        axis.text.y = element_text(size = 8, family = "mono"))

ggsave(file.path(output_dir, "speaker_utterance_distribution.png"),
       plot = p_utt, width = 14, height = 8, dpi = 300)
cat("Saved speaker utterance distribution plot.\n")

# ---------------------------------------------------------------------------
# 5. ADJ distribution by framing
# ---------------------------------------------------------------------------
llm_adj_main <- llm %>%
  count(model, framing, adj) %>%
  group_by(model, framing) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

human_adj_main <- human %>%
  count(framing, adj) %>%
  group_by(framing) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup() %>%
  mutate(model = "Human")

adj_combined <- bind_rows(
  human_adj_main %>% select(model, framing, adj, proportion),
  llm_adj_main %>% select(model, framing, adj, proportion)
) %>%
  mutate(
    adj     = factor(adj, levels = c("right", "wrong")),
    framing = factor(framing, levels = c("high", "low")),
    model   = factor(model, levels = c("Human", sort(unique(llm_adj_main$model))))
  )

p_adj <- ggplot(adj_combined, aes(x = adj, y = proportion, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ framing, nrow = 1,
             labeller = labeller(framing = c("high" = "High framing", "low" = "Low framing"))) +
  scale_fill_manual(values = main_palette, name = "Source") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(
    x     = "Adjective (ADJ)",
    y     = "Proportion",
    title = "Speaker Production: Adjective Choice (right vs wrong)",
    subtitle = "Human vs LLM. Do models use 'right' for high framing and 'wrong' for low?"
  ) +
  theme_csp() +
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "speaker_adj_distribution.png"),
       plot = p_adj, width = 10, height = 5, dpi = 300)
cat("Saved speaker ADJ distribution plot.\n")

# ---------------------------------------------------------------------------
# 6. Full sentence distribution (OQ x IQ x ADJ) - top sentences
# ---------------------------------------------------------------------------
human_sentences <- human %>%
  mutate(sentence = paste(oq, iq, adj, sep = " | ")) %>%
  count(framing, sentence) %>%
  group_by(framing) %>%
  mutate(proportion = n / sum(n)) %>%
  arrange(framing, desc(proportion)) %>%
  ungroup()

llm_sentences <- llm %>%
  mutate(sentence = paste(oq, iq, adj, sep = " | ")) %>%
  count(model, framing, sentence) %>%
  group_by(model, framing) %>%
  mutate(proportion = n / sum(n)) %>%
  arrange(model, framing, desc(proportion)) %>%
  ungroup()

# Save full distributions
write_csv(human_sentences, file.path(output_dir, "speaker_human_sentences.csv"))
write_csv(llm_sentences, file.path(output_dir, "speaker_llm_sentences.csv"))

# ---------------------------------------------------------------------------
# 7. Output format comparison: full_sentence vs three_blanks
#    Full utterance (OQ:IQ:ADJ) distribution split by format
# ---------------------------------------------------------------------------
llm_utt_fmt <- llm %>%
  mutate(utterance = paste0(oq, ":", iq, ":", adj)) %>%
  count(model, output_format, framing, utterance) %>%
  group_by(model, output_format, framing) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

# Use same top utterances from human data
fmt_data <- llm_utt_fmt %>%
  filter(utterance %in% top_utts) %>%
  mutate(
    utterance     = factor(utterance, levels = rev(top_utts)),
    framing       = factor(framing, levels = c("high", "low")),
    output_format = factor(output_format,
                           levels = c("full_sentence", "three_blanks"),
                           labels = c("Full sentence", "Three blanks"))
  ) %>%
  complete(model, output_format, framing, utterance, fill = list(proportion = 0))

p_format <- ggplot(fmt_data, aes(x = utterance, y = proportion, fill = output_format)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_grid(model ~ framing,
             labeller = labeller(framing = c("high" = "High framing",
                                             "low" = "Low framing"))) +
  scale_fill_manual(
    values = c("Full sentence" = project_colors[1], "Three blanks" = project_colors[3]),
    name = "Output format"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.05))) +
  coord_flip() +
  labs(
    x     = "Utterance (OQ : IQ : ADJ)",
    y     = "Proportion",
    title = "Output Format Comparison: Full Sentence vs Three Blanks",
    subtitle = "Does the output format affect the LLM's utterance choice?"
  ) +
  theme_csp() +
  theme(legend.position = "bottom",
        axis.text.y = element_text(size = 7, family = "mono"))

ggsave(file.path(output_dir, "speaker_format_comparison.png"),
       plot = p_format, width = 14, height = 10, dpi = 300)
cat("Saved output format comparison plot.\n")

# ---------------------------------------------------------------------------
# 8. Invalid rate summary
# ---------------------------------------------------------------------------
invalid_summary <- llm_raw %>%
  mutate(is_inv = is_valid == FALSE | is_valid == "False") %>%
  group_by(model, output_format) %>%
  summarise(
    total     = n(),
    n_invalid = sum(is_inv),
    invalid_rate = mean(is_inv),
    .groups = "drop"
  )

cat("\n--- Invalid rates ---\n")
print(invalid_summary)
write_csv(invalid_summary, file.path(output_dir, "speaker_invalid_rates.csv"))

# ---------------------------------------------------------------------------
# 9. Per-array comparison (item-level)
# ---------------------------------------------------------------------------
# Compare human and LLM distributions per array for the CogSci paper figure style
# This matches Figure 2 from the CogSci paper

# Human: compute per-array OQ x ADJ distribution
# Since human data uses different array sizes, we pool across sizes
# and match by the score pattern

cat("\n=== Speaker analysis complete ===\n")
