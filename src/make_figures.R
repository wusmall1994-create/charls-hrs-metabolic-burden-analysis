suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(dplyr)
  library(svglite)
  library(ragg)
})

root <- Sys.getenv("ANALYSIS_OUTPUT_DIR", unset = file.path(getwd(), "results"))
root <- normalizePath(root, winslash = "/", mustWork = TRUE)
out_dir <- Sys.getenv("FIGURE_OUTPUT_DIR", unset = file.path(root, "figures"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

COL_CHARLS <- "#1F5A7A"
COL_HRS <- "#D17A22"
COL_GRID <- "#D9DEE3"

theme_set(
  theme_classic(base_size = 8.3, base_family = "sans") +
    theme(
      axis.line.y = element_blank(),
      axis.line.x = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks.y = element_blank(),
      axis.ticks.x = element_line(linewidth = 0.35, colour = "black"),
      panel.grid.major.x = element_line(colour = COL_GRID, linewidth = 0.25),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 8),
      plot.title = element_text(size = 9, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 7.6, colour = "#4C545B"),
      plot.tag = element_text(size = 9, face = "bold"),
      plot.margin = margin(5, 7, 5, 5)
    )
)

save_pub_r <- function(plot, filename, width_mm = 183, height_mm = 105, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot)
  dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h, family = "sans")
  print(plot)
  dev.off()
  ragg::agg_tiff(paste0(filename, ".tiff"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()
  ragg::agg_png(paste0(filename, "_preview.png"), width = w, height = h, units = "in", res = 300)
  print(plot)
  dev.off()
}

raw <- read_csv(file.path(root, "charls_hrs_harmonized_comparison.csv"), show_col_types = FALSE)
labels <- c(
  persistent_low_depression = "Persistently low\ndepressive symptoms",
  memory_z = "Memory performance,\nper 1 SD higher"
)

long <- bind_rows(
  raw %>% transmute(outcome, variable, cohort = "CHARLS", effect = charls_effect,
                    low = charls_ci_low, high = charls_ci_high, difference_p = effect_difference_p),
  raw %>% transmute(outcome, variable, cohort = "HRS", effect = hrs_effect,
                    low = hrs_ci_low, high = hrs_ci_high, difference_p = effect_difference_p)
) %>%
  mutate(
    marker = unname(labels[variable]),
    marker = paste0(marker, "\nP[difference] = ", sprintf("%.3f", difference_p)),
    marker = factor(marker, levels = rev(unique(paste0(unname(labels[raw$variable]),
                                                   "\nP[difference] = ", sprintf("%.3f", raw$effect_difference_p)))))
  )

make_panel <- function(data, title, metric, limits, breaks, left_label, right_label) {
  ggplot(data, aes(x = effect, y = marker, colour = cohort, shape = cohort)) +
    geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.4, colour = "#6E7479") +
    geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0.10,
                  linewidth = 0.7, position = position_dodge(width = 0.44)) +
    geom_point(size = 2.5, stroke = 0.75, fill = "white",
               position = position_dodge(width = 0.44)) +
    scale_x_log10(limits = limits, breaks = breaks) +
    scale_colour_manual(values = c(CHARLS = COL_CHARLS, HRS = COL_HRS)) +
    scale_shape_manual(values = c(CHARLS = 16, HRS = 17)) +
    labs(
      x = paste0(metric, " (95% CI)"), y = NULL, title = title,
      subtitle = paste0(left_label, "  <-  1  ->  ", right_label)
    ) +
    guides(colour = guide_legend(override.aes = list(size = 2.5))) +
    theme(axis.text.y = element_text(size = 7.6, lineheight = 0.94))
}

p_a <- make_panel(
  filter(long, outcome == "Sustained independence"),
  "Sustained ADL independence", "Odds ratio", c(0.65, 3.35), c(0.75, 1, 1.5, 2, 3),
  "Lower odds", "Higher odds"
)
p_b <- make_panel(
  filter(long, outcome == "ADL disability vs independent"),
  "ADL disability versus independence", "Relative risk ratio", c(0.18, 1.45),
  c(0.2, 0.3, 0.5, 0.75, 1, 1.4), "Lower relative risk", "Higher relative risk"
)

fig1 <- p_a + p_b + plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") & theme(legend.position = "bottom")
save_pub_r(fig1, file.path(out_dir, "figure1_cross_cohort_forest"))

sens <- read_csv(file.path(root, "charls_hrs_continuous_depression_comparison.csv"), show_col_types = FALSE)
sens_long <- bind_rows(
  sens %>% transmute(outcome, cohort = "CHARLS", effect = charls_effect,
                     low = charls_ci_low, high = charls_ci_high, difference_p = effect_difference_p),
  sens %>% transmute(outcome, cohort = "HRS", effect = hrs_effect,
                     low = hrs_ci_low, high = hrs_ci_high, difference_p = effect_difference_p)
) %>%
  mutate(marker = paste0("Depressive burden, per 1 SD lower\nP[difference] = ",
                         sprintf("%.3f", difference_p)))

s_a <- make_panel(
  filter(sens_long, outcome == "Sustained independence"),
  "Sustained ADL independence", "Odds ratio", c(0.75, 1.85), c(0.8, 1, 1.25, 1.5, 1.75),
  "Lower odds", "Higher odds"
)
s_b <- make_panel(
  filter(sens_long, outcome == "ADL disability vs independent"),
  "ADL disability", "Relative risk ratio", c(0.38, 1.15),
  c(0.4, 0.5, 0.7, 1), "Lower risk", "Higher risk"
)
fig_s1 <- s_a + s_b + plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") & theme(legend.position = "bottom")
save_pub_r(fig_s1, file.path(out_dir, "supplementary_figure_s1_continuous_depression"), height_mm = 85)
