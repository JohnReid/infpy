#!/bin/bash

ARGS="-K 12 --min-iter 10 --max-iter 100 --tolerance 1e-2 --log-plot --seed 2 --point --num-starts 10"
#LD_LIBRARY_PATH=/usr/local/lib64/R/lib/:$LD_LIBRARY_PATH

data_sets="cshl sav"
weight_sets="weights inverse-weights"
stick_settings="0 1"
integrate_settings="0 1"

for p_values in $data_sets
do
	for weights in $weight_sets
	do
	    for stick in $stick_settings
	    do
	        if (( $stick ))
	        then
	        	stick_opt=--stick
	        	stick_name=s
	        else
	        	stick_opt=
	        	stick_name=d
	        fi
	    	for integrate in $integrate_settings
	    	do
		        if (( $integrate ))
		        then
		        	integrate_opt=--integrate
		        	integrate_name=i
		        else
		        	integrate_opt=
		        	integrate_name=f
		        fi
				name=$p_values-$weights-$stick_name$integrate_name
				python2.5 -O ../fit_beta_mixture.py $ARGS \
				    --log-file $name.log \
				    $stick_opt $integrate_opt \
					--x-file $p_values-p-values.txt \
					--predictions-file predictions-$name.txt \
					--weights-file lorenz-$weights.txt \
					--plot-file plot-$name.png
				#break
			done
		done
	done
	#break
done

