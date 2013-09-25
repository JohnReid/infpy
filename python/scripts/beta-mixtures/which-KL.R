
domain <- seq(-20, 20, length=1000)
p = .5 * dnorm(domain, -10, 3) + .5 * dnorm(domain, 10, 3)
q0 = dnorm(domain, -10, 3)
q1 = dnorm(domain, 0, 10)

grDevices::pdf(file='which-KL-0.pdf', width=2.33, height=1.5)
par(mar=c(0, 0, 0, 0)+.2)
plot(domain, p, type='l', xlab='', ylab='', xaxt='n', yaxt='n', ylim=c(0,max(q0)), ann=FALSE)
lines(domain, q0, type='l', lty='dashed')
grDevices::dev.off()

grDevices::pdf(file='which-KL-1.pdf', width=2.33, height=1.5)
par(mar=c(0, 0, 0, 0)+.2)
plot(domain, p, type='l', xlab='', ylab='', xaxt='n', yaxt='n', ylim=c(0,max(q0)), ann=FALSE)
lines(domain, q1, type='l', lty='dashed')
grDevices::dev.off()

