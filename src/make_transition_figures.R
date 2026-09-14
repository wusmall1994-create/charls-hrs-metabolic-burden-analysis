suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

root <- normalizePath(file.path(Sys.getenv("ANALYSIS_OUTPUT_DIR", "results"), "transition_upgrade"), winslash = "/", mustWork = FALSE)
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
  cohort = rep(c("CHARLS", "HRS", "ELSA"), c(4, 6, 5)),
  time = c(2011, 2015, 2018, 2020,
           2010, 2014, 2016, 2018, 2020, 2022,
           2008.5, 2012.5, 2014.5, 2016.5, 2018.5),
  kind = c("Biomarker 1", "Biomarker 2 + independent anchor", "ADL follow-up", "ADL follow-up",
           "Biomarker 1", "Biomarker 2 + independent anchor", "ADL follow-up", "ADL follow-up", "ADL follow-up", "ADL follow-up",
           "Biomarker 1", "Biomarker 2 + independent anchor", "ADL follow-up", "ADL follow-up", "ADL follow-up")
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

res <- read.csv(file.path(root, "enhanced_transition_models.csv"), stringsAsFactors = FALSE)
res <- res[res$variable %in% c("persistent_low_depression", "memory_z") &
             res$transition %in% c("incidence", "recovery") & res$model == "live" &
             res$threshold == "any_ADL" & res$cohort %in% c("CHARLS","HRS","ELSA"), ]
res$marker <- ifelse(res$variable == "persistent_low_depression", "Persistently low depressive symptoms", "Memory, per SD higher")
res$marker <- factor(res$marker, levels = c("Persistently low depressive symptoms", "Memory, per SD higher"))
res$panel <- ifelse(res$transition == "incidence", "A  Independence to ADL disability", "B  ADL disability to independence")
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
       caption = "Fully adjusted models use biomarker and inverse-probability response weights. Recovery was secondary.") +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom", strip.text = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0, size = 8))
save_three(p2, "Figure_2_transition_associations", height = 4.5)

flow <- data.frame(
  cohort=rep(c("CHARLS","HRS","ELSA"), c(4,6,4)),
  order=c(1:4,1:6,1:4),
  label=c("Exposure cohort: 653","Valid design: 648","2018 observed: 617","2020 observed: 580",
          "DBS overlap: 5,725","Four components: 4,743","Persistent burden: 1,800","Age >=50 + weight: 1,790","Independent anchor: 1,273","2016 observed: 1,225",
          "Exposure cohort: 897","Wave 7 observed: 830","Wave 8 observed: 746","Wave 9 observed: 670")
)
flow$cohort <- factor(flow$cohort, levels=c("CHARLS","HRS","ELSA"))
flow$final <- ave(flow$order, flow$cohort, FUN=function(x) x==max(x))
p3 <- ggplot(flow,aes(1,-order,label=label)) +
  geom_segment(data=subset(flow,!final),aes(xend=1,yend=-order-0.75),arrow=arrow(length=unit(0.08,"in")),color="#7A7A7A") +
  geom_label(fill="white",color="#16324F",linewidth=0.35,size=2.7,lineheight=.95) +
  facet_wrap(~cohort,scales="free",nrow=1) +
  scale_x_continuous(limits=c(0,2),expand=expansion(mult=.05)) +
  coord_cartesian(clip="on") + labs(x=NULL,y=NULL,caption="Counts show available construction stages and destination-state ascertainment.\nSource sampling structures differ before the harmonized exposure cohort.") +
  theme_void(base_family="Arial",base_size=9) + theme(strip.text=element_text(face="bold",size=11),plot.caption=element_text(hjust=0,size=8),plot.margin=margin(10,15,10,15))
save_three(p3, "Figure_S1_cohort_flow", height = 4.6)

