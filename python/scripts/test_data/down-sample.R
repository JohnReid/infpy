options(digits=22)
p_values = scan(file="cshl-p-values.txt")
near_1 = 9.99999999999e-1
p_values[p_values > near_1] = near_1
print(p_values[2])
cat(sample(p_values,  50), file="cshl-50.dat" , fill=80)
cat(sample(p_values, 100), file="cshl-100.dat", fill=80)
cat(sample(p_values, 300), file="cshl-300.dat", fill=80)
