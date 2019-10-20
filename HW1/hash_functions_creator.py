import random
import math
import os

################################################
num_hash_functions = 160
upper_bound_on_number_of_distinct_terms  = 261041*430
#upper_bound_on_number_of_distinct_terms =   138492
#upper_bound_on_number_of_distinct_terms =  3746518
path1= os.getcwd()
################################################


### primality checker
def is_prime(number):
	for j in range(2, int(math.sqrt(number)+1)):
		if (number % j) == 0: 
			return False
	return True



set_of_all_hash_functions = set()
while len(set_of_all_hash_functions) < num_hash_functions:
	a = random.randint(1, upper_bound_on_number_of_distinct_terms-1)
	b = random.randint(0, upper_bound_on_number_of_distinct_terms-1)
	p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
	while is_prime(p) == False:
		p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
	
	current_hash_function_id = tuple([a, b, p])
	set_of_all_hash_functions.add(current_hash_function_id)

#Storing hash_functions in .tsv file
f = open(path1+'/part_2/hash_functions/160.tsv', 'w')
f.write("a\tb\tp\tn"+"\n")
for a, b ,p in set_of_all_hash_functions:
	f.write(str(a) + "\t" + str(b) + "\t" + str(p) + "\t" + str(upper_bound_on_number_of_distinct_terms)+"\n")
f.close()
