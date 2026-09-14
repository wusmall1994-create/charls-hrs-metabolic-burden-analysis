suppressPackageStartupMessages(library(msm))

root <- file.path(Sys.getenv("ANALYSIS_OUTPUT_DIR", unset = "results"), "transition_upgrade")
pp <- read.csv(file.path(root, "person_period_internal.csv"), stringsAsFactors = FALSE)

make_long <- function(d) {
  base <- d[order(d$person_id, d$from_year), ]
  first <- base[!duplicated(base$person_id), ]
  x0 <- data.frame(
    person_id = first$person_id, year = first$from_year, state = first$origin,
    persistent_low_depression = first$persistent_low_depression,
    memory_z = first$memory_z, age = first$age, female = first$female,
    met_anchor = first$met_anchor, blood_weight = first$blood_weight
  )
  xd <- data.frame(
    person_id = d$person_id, year = d$to_year, state = d$destination,
    persistent_low_depression = d$persistent_low_depression,
    memory_z = d$memory_z, age = d$age, female = d$female,
    met_anchor = d$met_anchor, blood_weight = d$blood_weight
  )
  z <- unique(rbind(x0, xd))
  z <- z[order(z$person_id, z$year), ]
  z$time <- z$year - ave(z$year, z$person_id, FUN = min)
  z$age_z <- as.numeric(scale(z$age))
  z$met_z <- as.numeric(scale(z$met_anchor))
  z$state2 <- ifelse(z$state %in% c(0, 1), z$state + 1, NA)
  z
}

extract_hazards <- function(fit, cohort) {
  hz <- hazard.msm(fit)
  rows <- list()
  for (v in names(hz)) {
    mat <- hz[[v]]
    for (i in seq_len(nrow(mat))) {
      rn <- rownames(mat)[i]
      parts <- strsplit(gsub(" ", "", rn), "-")[[1]]
      rows[[length(rows) + 1]] <- data.frame(
        cohort = cohort, variable = v,
        transition = ifelse(parts[1] == "State1", "independence_to_disability", "disability_to_independence"),
        hazard_ratio = mat[i, 1], ci_low = mat[i, 2], ci_high = mat[i, 3]
      )
    }
  }
  do.call(rbind, rows)
}

all_results <- list()
diagnostics <- list()
for (cohort in unique(pp$cohort)) {
  z <- make_long(pp[pp$cohort == cohort, ])
  z <- z[!is.na(z$state2), ]
  first_state <- ave(z$state2, z$person_id, FUN = function(x) x[1])
  z <- z[first_state == 1, ]
  z <- z[complete.cases(z[, c("state2", "time", "persistent_low_depression", "memory_z", "age_z", "female", "met_z", "blood_weight")]), ]
  z$analysis_weight <- z$blood_weight / mean(z$blood_weight, na.rm = TRUE)
  qmat <- rbind(c(0, 0.10), c(0.20, 0))
  fit <- try(msm(
    state2 ~ time, subject = person_id, data = z, qmatrix = qmat,
    covariates = ~ persistent_low_depression + memory_z + age_z + female + met_z,
    subject.weights = analysis_weight, center = FALSE, control = list(maxit = 20000)
  ), silent = TRUE)
  if (inherits(fit, "try-error")) {
    diagnostics[[length(diagnostics) + 1]] <- data.frame(cohort = cohort, converged = FALSE, minus2loglik = NA, message = as.character(fit))
  } else {
    all_results[[length(all_results) + 1]] <- extract_hazards(fit, cohort)
    diagnostics[[length(diagnostics) + 1]] <- data.frame(
      cohort = cohort, converged = isTRUE(fit$opt$convergence == 0),
      minus2loglik = fit$minus2loglik, message = ""
    )
  }
}

results <- if (length(all_results)) do.call(rbind, all_results) else data.frame()
diag <- do.call(rbind, diagnostics)
write.csv(results, file.path(root, "multistate_sensitivity_results.csv"), row.names = FALSE)
write.csv(diag, file.path(root, "multistate_sensitivity_diagnostics.csv"), row.names = FALSE)
print(results)
print(diag)


