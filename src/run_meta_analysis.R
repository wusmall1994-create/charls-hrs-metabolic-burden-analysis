suppressPackageStartupMessages(library(metafor))
root <- file.path(Sys.getenv("ANALYSIS_OUTPUT_DIR", unset = "results"), "transition_upgrade")
d <- read.csv(file.path(root, "transition_mi_primary.csv"), stringsAsFactors = FALSE)
out <- list()
keys <- unique(d[, c("transition", "variable")])
for (i in seq_len(nrow(keys))) {
  q <- d[d$transition == keys$transition[i] & d$variable == keys$variable[i], ]
  if (nrow(q) < 2) next
  yi <- log(q$estimate)
  sei <- (log(q$ci_high) - log(q$ci_low)) / (2 * 1.96)
  fit <- rma.uni(yi = yi, sei = sei, method = "REML")
  out[[length(out) + 1]] <- data.frame(
    transition = keys$transition[i], variable = keys$variable[i], cohorts = nrow(q),
    estimate = exp(as.numeric(fit$b)), ci_low = exp(fit$ci.lb), ci_high = exp(fit$ci.ub),
    p_value = fit$pval, tau2 = fit$tau2, i2_percent = fit$I2, heterogeneity_p = fit$QEp
  )
}
res <- do.call(rbind, out)
write.csv(res, file.path(root, "transition_random_effects_meta.csv"), row.names = FALSE)
print(res)


