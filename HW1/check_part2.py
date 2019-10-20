import os
import datetime
import csv
import pandas as pd
import re
import pickle
print(datetime.datetime.now())
path1= os.getcwd()
# ----- method for reading pkl object -------------
def load_obj(name ):
    with open(path1 +"/part_2/"+ name + '.pkl', 'rb') as f:
        return pickle.load(f)


"""
Important memo: 
    * set working directory .\DMT4BaS_2019\HW_1\
    * Order of the execution of the files:
        1. execute hash_functions_creator.py
        2. execute part2.py
        3. execute check_part2.py

    * In command line
        > python part_2/sw/part2.py
        > python part_2/sw/hash_functions_creator.py
        > python part_2/sw/check_part2.py
"""
'''
Running java tool in the cmd:
java -Xmx3G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.88 8 20 ./hash_functions/160.tsv dict.tsv ./data/APPX_0804_088.tsv
With the above mentioned command; where JS is 0.88 we calculated near duplicate values.

With the above mentioned command; where JS is 0 we calculated the number of ALL CANDIDATE PAIRS:
java -Xmx3G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.0000 8 20 ./hash_functions/160.tsv dict.tsv ./data/APPX_0804.tsv

it would be the same as this Local Sensitive Hashing command:
java -Xmx3G tools.NearDuplicatesDetector lsh 8 20 ./hash_functions/160.tsv dict.tsv ./data/APPX_lsh.tsv

'''


# reading a dictionary from pkl file
Sh_dict=load_obj("ourdict")

# reading from a /data the file made by java tool
Near_d=pd.read_csv(path1+"/part_2/data/APPX_0804.tsv",sep='\t', usecols=['dummy_field','name_set_1','name_set_2'])
# renaming a columns
Near_d.columns=['js','d1','d2']

# reading from a /data the file made by java tool
Near_d_088=pd.read_csv(path1+"/part_2/data/APPX_0804_088.tsv",sep='\t', usecols=['dummy_field','name_set_1','name_set_2'])
# renaming a columns
Near_d_088.columns=['js','d1','d2']
print("the number of candidate near_dupl for t = 0.88 :  "+str(Near_d_088.shape[0]))


print("the number of candidate near_dupl:  "+str(Near_d.shape[0]))
#print("the number of candidate near_dupl:  "+str(Near_d_cand.shape[0]))

#------ nethod to calculate true Jaccard Similarity ---------
def j_s(s1,s2):
	return (len(s1.intersection(s2))/len(s1.union(s2)))
def getting_vals(data):
	tr_js=[] # the list where JS will be saved
	d1=list(data['d1'].values)  # the first members from each pair (list of the doc_ids of the first pair aka song ids)
	d2=list(data['d2'].values) # the second member from each pair (list of the doc_ids of the first pair aka song ids)
	for i in range(0, len(d1)):
		#if i%1000==0:
		#	print("1000")
		tr_js.append(j_s(set(Sh_dict[d1[i]]),set(Sh_dict[d2[i]]))) # adding to a list tru JS of each pair

	data['tr_js']=tr_js # adding true values of JS to dataframe
	js_tool=list(data['js'].values)  # getting a JS calculated by tool to make a comparison
	return (d1,d2,js_tool,tr_js)

n_088_d1, n_088_d2, js_t_088, tr_js_088= getting_vals(Near_d_088)
n_all_d1, n_all_d2, js_t_all, tr_js_all= getting_vals(Near_d)
#--------------------------------------
# the method below calculates false negatives number with threshold as a parameter
# we are looking if we give some value for threshold would the tool retrieve all the pairs with above JS that threshold
# so we checked in this method JS from tool and true JS calculated via our method,
# if they are both above the value of t we provided as a parameter,
# so it is TP, but if JS from tool is less, but the true value is more or equal to t so it is FN
def fals_neg_num(t,d1,d2,js_tool,tr_js):
	fn_pairs=[]
	fn=0
	for i in range(0, len(d1)):
		if js_tool[i]<t:
			if tr_js[i]>=t:
				fn=fn+1
				fn_pairs.append((d1[i],d2[i]))
	return (fn, fn_pairs)
# ----------------------------
# the method below calculates false positive number with threshold as a parameter
# in this method we checke when JS from tool is above t,
# but the true JS is not,
# so we consider it as FP
def fals_pos_num(t,d1,d2,js_tool,tr_js):
	fp_pairs=[]
	fp=0
	for i in range(0, len(d1)):
		if js_tool[i]>=t:
			if tr_js[i]<t:
				fp=fp+1
				fp_pairs.append((d1[i],d2[i]))
	return (fp, fp_pairs)


# for 0.88
fp_088_088, l_088=fals_pos_num(0.88, n_088_d1, n_088_d2, js_t_088, tr_js_088)
print("FP for t =0.88 : "+str(fp_088_088))	


# for all
fn_0_88, fn_list_0_88= fals_neg_num(0.88, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FN when t is 0.88: "+str(fn_0_88/Near_d.shape[0]))
fn_0_90, fn_list_0_90= fals_neg_num(0.9, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FN when t is 0.9: "+ str(fn_0_90/Near_d.shape[0]))
fn_0_95, fn_list_0_95= fals_neg_num(0.95, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FN when t is 0.95: "+str(fn_0_95/Near_d.shape[0]))
fn_1, fn_list_1= fals_neg_num(1.0, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FN when t is 1: "+str(fn_1/Near_d.shape[0]))

fp_085, fp_l_085=fals_pos_num(0.85, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.85: "+ str(fp_085/Near_d.shape[0]))
fp_080, fp_l_080=fals_pos_num(0.80, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.80: "+ str(fp_080/Near_d.shape[0]))
fp_075, fp_l_075=fals_pos_num(0.75, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.75: "+ str(fp_075/Near_d.shape[0]))
fp_070, fp_l_070=fals_pos_num(0.70, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.70: "+ str(fp_070/Near_d.shape[0]))
fp_065, fp_l_065=fals_pos_num(0.65, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.65: "+str(fp_065/Near_d.shape[0]))
fp_060, fp_l_060=fals_pos_num(0.60, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.60: "+str(fp_060/Near_d.shape[0]))
fp_055, fp_l_055=fals_pos_num(0.55, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.55: "+str(fp_055/Near_d.shape[0]))
fp_050, fp_l_050=fals_pos_num(0.50, n_all_d1, n_all_d2, js_t_all, tr_js_all)
print("the probability of FP when t is 0.5: "+str(fp_050/Near_d.shape[0]))



print(datetime.datetime.now())