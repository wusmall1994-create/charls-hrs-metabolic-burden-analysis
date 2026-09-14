suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

root <- file.path(Sys.getenv("ANALYSIS_OUTPUT_DIR", unset = "results"), "transition_upgrade")
figdir <- file.path(root, "figures")
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)
theme_set(theme_minimal(base_family = "Arial", base_size = 9))
cols <- c("Persistently low depressive symptoms" = "#0072B2", "Memory, per SD higher" = "#D55E00")

save_three <- function(plot, stem, width = 7.09, height = 4.8) {
  ggsave(file.path(figdir, paste0(stem, ".pdf")), plot, width = width, height = height, units = "in", device = cairo_pdf)
  ggsave(file.path(figdir, paste0(stem, ".png")), plot, width = width, height = height, units = "in", dpi = 220, bg = "white")
  ggsave(file.path(figdir, paste0(stem, ".tiff")), plot, width = width, height = height, units = "in", dpi = 600,
         compression = "lzw", bg = "white")
}

timeline <- data.frame(
  cohort = rep(c("CHARLS", "HRS", "ELSA"), c(3, 5, 4)),
  time = c(2011, 2015, 2018, 2010, 2014, 2016, 2018, 2022, 2008.5, 2012.5, 2014.5, 2018.5),
  kind = c("Biomarker 1", "Biomarker 2 + independent anchor", "ADL follow-up",
           "Biomarker 1", "Biomarker 2 + independent anchor", "ADL follow-up", "ADL follow-up", "ADL follow-up",
           "Biomarker 1", "Biomarker 2 + independent anchor", "ADL follow-up", "ADL follow-up")
)
timeline$cohort <- factor(timeline$cohort, levels = c("CHARLS", "HRS", "ELSA"))
segments <- data.frame(cohort = factor(c("CHARLS", "HRS", "ELSA"), levels = levels(timeline$cohort)),
                       start = c(2011, 2010, 2008.5), end = c(2020, 2022, 2018.5))
p1 <- ggplot(timeline, aes(time, cohort)) +
  geom_segment(data = segments, aes(x = start, xend = end, y = cohort, yend = cohort), inherit.aes = FALSE,
               linewidth = 0.7, color = "#666666") +
  geom_point(aes(shape = kind, fill = kind), size = 3.2, color = "#222222") +
  scale_shape_manual(values = c("Biomarker 1" = 21, "Biomarker 2 + independent anchor" = 23, "ADL follow-up" = 22)) +
  scale_fill_manual(values = c("Biomarker 1" = "#56B4E9", "Biomarker 2 + independent anchor" = "#009E73", "ADL follow-up" = "white")) +
  scale_x_continuous(breaks = seq(2008, 2022, 2), limits = c(2007.5, 2023)) +
  labs(x = "Calendar year", y = NULL, shape = NULL, fill = NULL,
       caption = "ELSA Wave 4 and Wave 6 occurred in 2008/09 and 2012/13. Death-transition analyses use CHARLS and HRS only.") +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        legend.position = "bottom", plot.caption = element_text(hjust = 0, size = 8))
save_three(p1, "Figure_1_transition_design", height = 3.4)

res <- read.csv(file.path(root, "transition_mi_primary.csv"), stringsAsFactors = FALSE)
res <- res[res$variable %in% c("persistent_low_depression", "memory_z") &
             res$transition %in% c("independence_to_disability", "disability_to_independence"), ]
res$marker <- ifelse(res$variable == "persistent_low_depression", "Persistently low depressive symptoms", "Memory, per SD higher")
res$marker <- factor(res$marker, levels = c("Persistently low depressive symptoms", "Memory, per SD higher"))
res$panel <- ifelse(res$transition == "independence_to_disability", "A  Independence to ADL disability", "B  ADL disability to independence")
res$cohort <- factor(res$cohort, levels = c("CHARLS", "HRS", "ELSA"))
res$label <- paste0(res$cohort, "  (", res$events, "/", res$records, ")")
res$label <- factor(res$label, levels = rev(unique(res$label)))
p2 <- ggplot(res, aes(estimate, label, color = marker, shape = marker)) +
  geom_vline(xintercept = 1, linewidth = 0.45, linetype = 2, color = "#666666") +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0, linewidth = 0.55,
                 position = position_dodge(width = 0.48)) +
  geom_point(size = 2.4, position = position_dodge(width = 0.48)) +
  facet_wrap(~panel, scales = "free_y", ncol = 2) +
  scale_x_log10(breaks = c(0.25, 0.5, 1, 2, 4, 8), limits = c(0.2, 12)) +
  scale_color_manual(values = cols) + scale_shape_manual(values = c(16, 17)) +
  labs(x = "Adjusted odds ratio (log scale)", y = "Cohort (events / transition records)", color = NULL, shape = NULL,
       caption = "Models include both markers and adjust for age, sex, metabolic-abnormality count, and interval.\nRecovery was secondary.") +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom", strip.text = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0, size = 8))
save_three(p2, "Figure_2_transition_associations", height = 4.5)

iv <- read.csv(file.path(root, "transition_interval_specific.csv"), stringsAsFactors = FALSE)
iv$marker <- ifelse(iv$variable == "persistent_low_depression", "Persistently low depressive symptoms", "Memory, per SD higher")
iv$interval <- paste(iv$cohort, paste0(iv$from_year, "-", iv$to_year), sep = ": ")
iv$interval <- factor(iv$interval, levels = rev(unique(iv$interval)))
p3 <- ggplot(iv, aes(estimate, interval, color = marker)) +
  geom_vline(xintercept = 1, linewidth = 0.45, linetype = 2, color = "#666666") +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0, linewidth = 0.5) +
  geom_point(size = 2.1) +
  facet_wrap(~marker, ncol = 2) +
  scale_x_log10(breaks = c(0.2, 0.5, 1, 2)) + scale_color_manual(values = cols, guide = "none") +
  labs(x = "Adjusted odds ratio (log scale)", y = NULL,
       caption = "Complete-case interval-specific sensitivity models; intervals are not independent replications.") +
  theme(panel.grid.minor = element_blank(), strip.text = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0, size = 8))
save_three(p3, "Figure_S1_interval_stability", height = 5.2)


