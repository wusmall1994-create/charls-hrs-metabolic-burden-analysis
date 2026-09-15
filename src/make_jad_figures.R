suppressPackageStartupMessages({
  library(ggplot2)
  library(metafor)
  library(grid)
})

output_root <- Sys.getenv("ANALYSIS_OUTPUT_DIR", unset = "results")
root <- file.path(output_root, "jad_submission")
source_root <- file.path(output_root, "transition_upgrade")
figdir <- file.path(root, "figures")
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)
theme_set(theme_minimal(base_family = "Arial", base_size = 9))

save_three <- function(plot, stem, width = 7.09, height = 4.4) {
  ggsave(file.path(figdir, paste0(stem, ".pdf")), plot, width = width, height = height,
         units = "in", device = cairo_pdf)
  ggsave(file.path(figdir, paste0(stem, ".png")), plot, width = width, height = height,
         units = "in", dpi = 300, bg = "white")
  ggsave(file.path(figdir, paste0(stem, ".tiff")), plot, width = width, height = height,
         units = "in", dpi = 600, compression = "lzw", bg = "white")
}

primary <- read.csv(file.path(source_root, "jad_primary_models.csv"), stringsAsFactors = FALSE)
timing <- read.csv(file.path(source_root, "jad_covariate_timing.csv"), stringsAsFactors = FALSE)
spline <- read.csv(file.path(source_root, "jad_spline_curves.csv"), stringsAsFactors = FALSE)
spline_tests <- read.csv(file.path(source_root, "jad_spline_nonlinearity.csv"), stringsAsFactors = FALSE)
interval_sensitivity <- read.csv(file.path(source_root, "jad_interval_sensitivity.csv"), stringsAsFactors = FALSE)
atten <- read.csv(file.path(source_root, "jad_attenuation_matrix.csv"), stringsAsFactors = FALSE)

write.csv(primary, file.path(root, "jad_primary_models.csv"), row.names = FALSE)
write.csv(timing, file.path(root, "jad_covariate_timing.csv"), row.names = FALSE)
write.csv(spline, file.path(root, "jad_spline_curves.csv"), row.names = FALSE)
write.csv(spline_tests, file.path(root, "jad_spline_nonlinearity.csv"), row.names = FALSE)
write.csv(interval_sensitivity, file.path(root, "jad_interval_sensitivity.csv"), row.names = FALSE)
write.csv(atten, file.path(root, "jad_attenuation_matrix.csv"), row.names = FALSE)

meta_rows <- list()
for (a in unique(primary$analysis)) {
  q <- primary[primary$analysis == a, ]
  yi <- log(q$estimate)
  sei <- (log(q$ci_high) - log(q$ci_low)) / (2 * 1.96)
  fit <- rma.uni(yi = yi, sei = sei, method = "REML")
  meta_rows[[length(meta_rows) + 1]] <- data.frame(
    analysis = a, estimate = exp(as.numeric(fit$b)), ci_low = exp(fit$ci.lb),
    ci_high = exp(fit$ci.ub), p_value = fit$pval, i2_percent = fit$I2,
    heterogeneity_p = fit$QEp
  )
}
meta <- do.call(rbind, meta_rows)
write.csv(meta, file.path(root, "jad_random_effects_meta.csv"), row.names = FALSE)

primary$marker <- factor(primary$analysis,
  levels = c("principal_continuous_first_health", "binary_first", "memory_firsthealth"),
  labels = c("Depressive symptom burden, per SD higher",
             "Persistently low depressive symptoms",
             "Memory, per SD higher"))
primary$cohort <- factor(primary$cohort, levels = c("CHARLS", "HRS", "ELSA"))
primary$label <- factor(primary$cohort, levels = rev(levels(primary$cohort)))
p1 <- ggplot(primary, aes(estimate, label, color = marker, shape = marker)) +
  geom_vline(xintercept = 1, linewidth = 0.45, linetype = 2, color = "#666666") +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0, linewidth = 0.55,
                 position = position_dodge(width = 0.55)) +
  geom_point(size = 2.5, position = position_dodge(width = 0.55)) +
  scale_x_log10(breaks = c(0.4, 0.6, 1, 1.5, 2), limits = c(0.32, 2.2)) +
  scale_color_manual(values = c("#D55E00", "#0072B2", "#009E73")) +
  scale_shape_manual(values = c(16, 17, 15)) +
  labs(x = "Adjusted odds ratio for independence-to-disability transition (log scale)",
       y = NULL, color = NULL, shape = NULL,
       caption = "Higher continuous symptom burden indicates worse symptoms. Models use first-wave health covariates and response weights.") +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        legend.position = "bottom", plot.caption = element_text(hjust = 0, size = 8))
save_three(p1, "Figure_1_primary_associations", height = 4.2)

spline$cohort <- factor(spline$cohort, levels = c("CHARLS", "HRS", "ELSA"))
spline_tests$cohort <- factor(spline_tests$cohort, levels = c("CHARLS", "HRS", "ELSA"))
spline_tests$label <- sprintf("P for nonlinearity = %.3f", spline_tests$p_nonlinearity)
p_locations <- merge(
  aggregate(burden_z ~ cohort, spline, min),
  aggregate(ci_high ~ cohort, spline, max), by = "cohort"
)
names(p_locations)[names(p_locations) == "burden_z"] <- "x"
names(p_locations)[names(p_locations) == "ci_high"] <- "y"
p_locations <- merge(p_locations, spline_tests[c("cohort", "label")], by = "cohort")
p2 <- ggplot(spline, aes(burden_z, estimate)) +
  geom_hline(yintercept = 1, linetype = 2, linewidth = 0.4, color = "#666666") +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), fill = "#56B4E9", alpha = 0.25) +
  geom_line(color = "#0072B2", linewidth = 0.8) +
  geom_text(data = p_locations, aes(x = x, y = y, label = label),
            inherit.aes = FALSE, hjust = 0, vjust = 1.2, size = 2.7) +
  facet_wrap(~cohort, nrow = 1) +
  scale_y_log10(breaks = c(0.5, 0.75, 1, 1.5, 2, 3)) +
  labs(x = "Two-wave depressive symptom burden (cohort-standardized SD)",
       y = "Adjusted odds ratio (reference = cohort mean)",
       caption = paste0("Restricted cubic splines use knots at the 5th, 35th, 65th, and 95th percentiles.\n",
                        "Shaded bands are 95% confidence intervals; P values are Rubin-pooled Wald tests of the nonlinear terms.")) +
  theme(panel.grid.minor = element_blank(), strip.text = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0, size = 8))
save_three(p2, "Figure_2_exposure_response", height = 3.7)

stage_labels <- c(marker_interval = "Marker + interval", demographic = "+ Demographic",
                  behavior = "+ Behavior", health = "+ First-wave health", full = "+ Metabolic count")
qa <- atten[atten$exposure == "depression_burden_z" & tolower(as.character(atten$response_weighted)) == "true", ]
qa$stage <- factor(qa$stage, levels = names(stage_labels), labels = stage_labels)
qa$cohort <- factor(qa$cohort, levels = c("CHARLS", "HRS", "ELSA"))
p3 <- ggplot(qa, aes(stage, estimate, group = cohort, color = cohort)) +
  geom_hline(yintercept = 1, linetype = 2, linewidth = 0.4, color = "#777777") +
  geom_line(linewidth = 0.65) + geom_point(size = 2) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.08, linewidth = 0.4) +
  scale_color_manual(values = c("#0072B2", "#D55E00", "#009E73")) +
  labs(x = NULL, y = "Odds ratio per SD higher symptom burden", color = NULL,
       caption = "All models include interview-interval indicators and inverse-probability response weights.") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1), panel.grid.minor = element_blank(),
        legend.position = "bottom", plot.caption = element_text(hjust = 0, size = 8))
save_three(p3, "Figure_S2_attenuation", height = 4.0)

timeline <- data.frame(
  cohort = rep(c("CHARLS", "HRS", "ELSA"), c(4, 6, 5)),
  time = c(2011, 2015, 2018, 2020, 2010, 2014, 2016, 2018, 2020, 2022,
           2008.5, 2012.5, 2014.5, 2016.5, 2018.5),
  kind = c("Biomarker 1", "Biomarker 2 + independent anchor", "ADL follow-up", "ADL follow-up",
           "Biomarker 1", "Biomarker 2 + independent anchor", rep("ADL follow-up", 4),
           "Biomarker 1", "Biomarker 2 + independent anchor", rep("ADL follow-up", 3)))
timeline$cohort <- factor(timeline$cohort, levels = c("CHARLS", "HRS", "ELSA"))
segments <- data.frame(cohort = factor(c("CHARLS", "HRS", "ELSA"), levels = levels(timeline$cohort)),
                       start = c(2011, 2010, 2008.5), end = c(2020, 2022, 2018.5))
p4 <- ggplot(timeline, aes(time, cohort)) +
  geom_segment(data = segments, aes(x = start, xend = end, y = cohort, yend = cohort),
               inherit.aes = FALSE, linewidth = 0.7, color = "#666666") +
  geom_point(aes(shape = kind, fill = kind), size = 3.2, color = "#222222") +
  scale_shape_manual(values = c("Biomarker 1" = 21, "Biomarker 2 + independent anchor" = 23, "ADL follow-up" = 22)) +
  scale_fill_manual(values = c("Biomarker 1" = "#56B4E9", "Biomarker 2 + independent anchor" = "#009E73", "ADL follow-up" = "white")) +
  scale_x_continuous(breaks = seq(2008, 2022, 2), limits = c(2007.5, 2023)) +
  labs(x = "Calendar year", y = NULL, shape = NULL, fill = NULL) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), legend.position = "bottom")
save_three(p4, "Figure_S1_design", height = 3.2)

print(meta)
