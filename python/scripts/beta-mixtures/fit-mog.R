
library(mclust)

sigmoid <- function(x){
	return (1 / (1 + exp(-x)))
}

inv_sigmoid <- function(x){
	return (log(x) - log(1-x))
}

p = scan(file="../scripts/test_data/cshl-p-values.txt")

transformed_p = inv_sigmoid(p)

#kde = density(transformed_p)
#plot(kde)

clustering <- Mclust(transformed_p)
domain <- seq(-100, 100, by=.1)
estimate <- dens(
	modelName=clustering$modelName,
	data=domain,
	parameters=clustering$parameters
)
grDevices::pdf(file='mog-distribution.pdf', width=4.652, height=1.6)
par(mar=c(2, 4, 0, 0)+.2)
mclust1Dplot(transformed_p, clustering$parameters, what = c("density"), identify=FALSE)
if( length(transformed_p) > 500 ) {
    ruged_transformed_p = sample(transformed_p, 500)
	rug(ruged_transformed_p)
} else {
	rug(transformed_p)
}
grDevices::dev.off()

grDevices::pdf(file='mog-01.pdf', width=4.652, height=1.6)
par(mar=c(2, 4, 0, 0)+.2)
#kde = density(p, from=0, to=1)
#plot(kde)
plot(sigmoid(domain), estimate, type='l', xlab='', ylab='density')
if( length(p) > 500 ) {
	rug(sample(p, 500))
} else {
	rug(p)
}
grDevices::dev.off()
