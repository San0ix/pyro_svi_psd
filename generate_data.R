library("beyondWhittle")
library("TSA")

set.seed(6578)

# export the sunspots dataset
dataset <- "sunspots"
data <- sqrt(as.numeric(sunspot.year))
data <- data - mean(data)
write.csv(data, file = "data/sunspots.csv", row.names = FALSE)
write.csv(periodogram(data, plot = FALSE)$spec, file = "data/sunspots_periodogram.csv", row.names = FALSE)
write.csv(periodogram(data, plot = FALSE)$freq * 2 * pi, file = "data/sunspots_frequency.csv", row.names = FALSE)

# run mcmc and export the results
mcmc <- gibbs_np(data = data, Ntotal = 10000, burnin = 4000, thin = 1)
write.csv(mcmc$psd.median, file = paste0("data/", dataset, "_mcmc.csv"), row.names = FALSE)


# generate and export an MA time series
dataset <- "ma_simulation"
n <- 256
ma <- c(0, 0, 0, 0, 0, 0, 0, 0.95)
data <- arima.sim(n = n, model = list(ma = ma))
omega <- fourier_freq(n)
psd_true <- psd_arma(omega, ar = numeric(0), ma = ma, sigma2 = 1)
write.csv(data, file = "data/ma_simulation.csv", row.names = FALSE)
write.csv(periodogram(data, plot = FALSE)$spec, file = "data/ma_simulation_periodogram.csv", row.names = FALSE)
write.csv(periodogram(data, plot = FALSE)$freq * 2 * pi, file = "data/ma_simulation_frequency.csv", row.names = FALSE)
write.csv(psd_true, file = "data/ma_simulation_true_psd.csv", row.names = FALSE)

# run mcmc and export the results
mcmc <- gibbs_np(data = data, Ntotal = 10000, burnin = 4000, thin = 1)
write.csv(mcmc$psd.median, file = paste0("data/", dataset, "_mcmc.csv"), row.names = FALSE)
