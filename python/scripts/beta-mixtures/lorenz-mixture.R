
domain <- seq(0, 1, length=1000)
true <- .2 * dbeta(domain, 1, 50) + .1 * dbeta(domain, 1, 1.5)
false <- .7 * (.1 * dbeta(domain, 30, 1) + .9)

grDevices::pdf(file='lorenz-mixture.pdf', width=4.652, height=2.2)
par(mar=c(4, 4, 0, 0)+.2)
plot(domain, true + false, type='l', xlab='p-value', ylab='density', ylim=c(0,max(true+false)))
lines(domain, false, type='l', lty='dashed', col='red')
lines(domain, true , type='l', lty='dashed', col='blue')
grDevices::dev.off()
