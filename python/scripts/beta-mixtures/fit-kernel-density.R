
options(digits=22)
x = scan(file="../scripts/test_data/cshl-p-values.txt")
near_1 = 9.99999999999e-1
#p_values[p_values > near_1] = near_1
density = density(x, from=0, to=1, adjust=.15)
grDevices::pdf(file='kernel-density.pdf', width=4.652, height=3.2)
par(mar=c(4, 4, 0, 0)+.2)
plot(density, ty="l", main="", log="y")
if( length(x) > 500 ) {
	rug(sample(x, 500))
} else {
	rug(x)
}
grDevices::dev.off()