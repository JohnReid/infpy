domain_size = 1000
p = 1:domain_size / domain_size
beta_weight = .1
beta = beta_weight * dbeta(p, .6, 10)
#beta[0:30] = NA
uniform = 1. - beta_weight
dist = beta + uniform
grDevices::pdf(file='bum-distribution.pdf', width=4.652, height=3.2)
par(mar=c(4, 4, 0, 0)+.2)
plot(p, dist, type='l', ylim=c(0, dist[.003*domain_size]), xlab='p-value', ylab='density')
lines(c(0, 1), c(1. - beta_weight, 1. - beta_weight), lty="dashed")
lines(c(.05, .05), c(0, dist[.05*domain_size]), lty="dashed")
grDevices::dev.off()