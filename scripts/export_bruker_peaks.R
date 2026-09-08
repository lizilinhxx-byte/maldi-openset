#!/usr/bin/env Rscript

try(Sys.setlocale("LC_ALL", "English_United States.utf8"), silent = TRUE)

suppressPackageStartupMessages({
  library(MALDIquant)
  library(MALDIquantForeign)
  library(jsonlite)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: export_bruker_peaks.R INPUT_DIR OUTPUT_CSV OUTPUT_MANIFEST_JSON [CONFIG_JSON] [PROFILE_FLOAT32]")
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_csv <- args[[2]]
output_manifest <- args[[3]]
config <- if (length(args) >= 4) fromJSON(args[[4]]) else list()
profile_matrix <- if (length(args) >= 5) args[[5]] else sub("\\.csv$", ".float32", output_csv)

mass_min <- ifelse(is.null(config$mass_min), 2000, config$mass_min)
mass_max <- ifelse(is.null(config$mass_max), 13000, config$mass_max)
half_window <- ifelse(is.null(config$preprocessing$half_window_size), 10, config$preprocessing$half_window_size)
baseline_iterations <- ifelse(is.null(config$preprocessing$baseline_iterations), 100, config$preprocessing$baseline_iterations)
peak_snr <- ifelse(is.null(config$preprocessing$peak_snr), 2, config$preprocessing$peak_snr)
peak_half_window <- ifelse(is.null(config$preprocessing$peak_half_window_size), 10, config$preprocessing$peak_half_window_size)
bin_width <- ifelse(is.null(config$bin_width), 1, config$bin_width)
n_bins <- as.integer(ceiling((mass_max - mass_min) / bin_width))

fid_files <- list.files(input_dir, pattern = "^fid$", recursive = TRUE, full.names = TRUE)
if (length(fid_files) == 0) stop("no Bruker fid files found")
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)

header <- data.frame(
  spectrum_id = character(), source_path = character(), scan_index = integer(),
  mass = numeric(), intensity = numeric(), stringsAsFactors = FALSE
)
write.table(header, output_csv, sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
profile_connection <- file(profile_matrix, open = "wb")
on.exit(close(profile_connection), add = TRUE)

records <- list()
errors <- list()
counter <- 0L

process_fid <- function(fid) {
  relative_path <- substring(normalizePath(fid, winslash = "/"), nchar(normalizePath(input_dir, winslash = "/")) + 2)
  imported <- tryCatch(
    suppressWarnings(importBrukerFlex(fid, massRange = c(mass_min, mass_max), verbose = FALSE)),
    error = function(e) e
  )
  if (inherits(imported, "error")) {
    return(list(path = relative_path, error = conditionMessage(imported), scans = list()))
  }
  scans <- lapply(seq_along(imported), function(scan_index) {
    spectrum <- imported[[scan_index]]
    spectrum <- transformIntensity(spectrum, method = "sqrt")
    spectrum <- smoothIntensity(spectrum, method = "SavitzkyGolay", halfWindowSize = half_window)
    spectrum <- removeBaseline(spectrum, method = "SNIP", iterations = baseline_iterations)
    processed_intensity <- pmax(intensity(spectrum), 0)
    bin_index <- floor((mass(spectrum) - mass_min) / bin_width) + 1L
    keep <- is.finite(processed_intensity) & bin_index >= 1L & bin_index <= n_bins
    binned <- numeric(n_bins)
    if (any(keep)) {
      aggregated <- rowsum(processed_intensity[keep], group = bin_index[keep], reorder = FALSE)
      binned[as.integer(rownames(aggregated))] <- aggregated[, 1]
    }
    tic <- sum(binned)
    if (is.finite(tic) && tic > 0) binned <- binned / tic
    peaks <- detectPeaks(spectrum, method = "MAD", SNR = peak_snr, halfWindowSize = peak_half_window)
    list(
      scan_index = scan_index,
      mass = mass(peaks),
      intensity = intensity(peaks),
      raw_point_count = length(mass(spectrum)),
      min_mass = ifelse(length(mass(spectrum)), min(mass(spectrum)), NA),
      max_mass = ifelse(length(mass(spectrum)), max(mass(spectrum)), NA),
      binned = binned
    )
  })
  list(path = relative_path, error = NULL, scans = scans)
}

workers <- suppressWarnings(as.integer(Sys.getenv("MALDI_WORKERS", unset = "8")))
workers <- max(1L, min(workers, max(1L, detectCores() - 1L)))
cluster <- if (workers > 1L) makeCluster(workers) else NULL
if (!is.null(cluster)) {
  clusterEvalQ(cluster, suppressPackageStartupMessages({library(MALDIquant); library(MALDIquantForeign)}))
  clusterExport(
    cluster,
    c("input_dir", "mass_min", "mass_max", "bin_width", "n_bins", "half_window", "baseline_iterations", "peak_snr", "peak_half_window", "process_fid"),
    envir = environment()
  )
  on.exit(stopCluster(cluster), add = TRUE)
}

fid_files <- sort(fid_files)
chunk_size <- 250L
for (start in seq(1L, length(fid_files), by = chunk_size)) {
  selected <- fid_files[start:min(start + chunk_size - 1L, length(fid_files))]
  batch <- if (is.null(cluster)) lapply(selected, process_fid) else parLapply(cluster, selected, process_fid)
  peak_frames <- list()
  profile_rows <- list()
  for (item in batch) {
    if (!is.null(item$error)) {
      errors[[length(errors) + 1L]] <- list(path = item$path, error = item$error)
      next
    }
    for (scan in item$scans) {
      counter <- counter + 1L
      spectrum_id <- sprintf("raw_%06d", counter)
      if (length(scan$mass) > 0) {
        peak_frames[[length(peak_frames) + 1L]] <- data.frame(
          spectrum_id = spectrum_id, source_path = item$path, scan_index = scan$scan_index,
          mass = scan$mass, intensity = scan$intensity, stringsAsFactors = FALSE
        )
      }
      records[[length(records) + 1L]] <- list(
        spectrum_id = spectrum_id, source_path = item$path, scan_index = scan$scan_index,
        peak_count = length(scan$mass), raw_point_count = scan$raw_point_count,
        min_mass = scan$min_mass, max_mass = scan$max_mass
      )
      profile_rows[[length(profile_rows) + 1L]] <- scan$binned
    }
  }
  if (length(peak_frames)) {
    write.table(
      do.call(rbind, peak_frames), output_csv, sep = ",", row.names = FALSE,
      col.names = FALSE, quote = TRUE, append = TRUE
    )
  }
  if (length(profile_rows)) {
    writeBin(as.numeric(unlist(profile_rows, use.names = FALSE)), profile_connection, size = 4L, endian = "little")
  }
  message("processed ", counter, " spectra from ", min(start + chunk_size - 1L, length(fid_files)), " files")
}

write_json(
  list(
    input_dir = input_dir,
    generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
    parameters = list(
      mass_min = mass_min, mass_max = mass_max, half_window_size = half_window,
      baseline_iterations = baseline_iterations, peak_snr = peak_snr,
      peak_half_window_size = peak_half_window, bin_width = bin_width, n_bins = n_bins
    ),
    profile_matrix = list(path = normalizePath(profile_matrix, winslash = "/", mustWork = FALSE), dtype = "float32-little-endian", n_bins = n_bins),
    spectrum_count = length(records), error_count = length(errors),
    records = records, errors = errors,
    session_info = capture.output(sessionInfo())
  ),
  output_manifest,
  pretty = TRUE,
  auto_unbox = TRUE,
  na = "null"
)

if (length(errors) > 0) warning(length(errors), " fid files could not be imported")
