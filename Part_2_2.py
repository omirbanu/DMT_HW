# =============================================================================
# PART 2.2
# =============================================================================
import os
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
start=datetime.datetime.now()
print(start)
# in order to execute, please,
# put a dir in cmd to execute this file : ./DMT4BaS_2019/HW_2

# then type following:
# python Part_2_2/sw/Part_2_2.py

path1= os.getcwd()
given_graph_edges=[]
#---------------------------------------------
#reading edges from file

with open(path1+"\\Part_2_2\\dataset\\Base_Set___UserID__ItemID__PART_2_2.tsv") as f:
	for line in f:
		given_graph_edges.append(tuple(map(int, line.rstrip('\n').split('\t'))))

# make a dictionary where keys = users, values = items

user_item_dict={'a':{'a','b'}}
for (u,v) in given_graph_edges:
    if u in user_item_dict.keys():
        temp=user_item_dict[u]
        temp.add(v)
        user_item_dict[u].update(temp)
    else:
        user_item_dict[u]={v}
del user_item_dict['a']

# ------------------------------------------------------
# Personalized  Pagerank
# reading a file

PR_dict=defaultdict(set)
s=0
with open(path1+"\\Part_2_2\\dataset\\ItemID__PersonalizedPageRank_Vector.tsv") as f:
	for line in f:
		s=s+1
		
		some_line=line.split('\t')
		PR_dict[int(some_line[0])]=[tuple(map(float,x.replace("[",'').replace('(','').replace(')','').replace('\n','').replace(']','').split(','))) for x in some_line[1].split('), (')]
		
# -----------------------------------------------------------
# Ground truth # reading a file
GT=defaultdict(set)
g_t=list()
with open(path1+"\\Part_2_2\\dataset\\Ground_Truth___UserID__ItemID__PART_2_2.tsv") as f:
	for line in f:
		g_t.append(tuple(map(int, line.rstrip('\n').split('\t'))))

for u, i in g_t:
	temp=set()
	if u in GT.keys():
		temp=set(GT[u])
		temp.add(i)
		GT[u]=temp
	else:
		GT[u]=set([i])
# -----------------------------------------------------------------------
# the function below return recomended set of items for one user
# ***************************************
def calc_recom1(S, user):
    s_len=len(S)
    NN=len(GT[user])
    values=np.zeros(len(PR_dict))
    for s in S:
        pr_items=[]
        pr_prs=[]
        pr_v= PR_dict[s] # getting pr-values for one item in set of items of given user_id
        for i, r in pr_v:
            pr_items.append(i) # put items in a list of items
            pr_prs.append(r) # put rankings of items to another list
        x=np.array(pr_prs)
        values=values+x*1/s_len # summing a vector with another vector
    answ=values.tolist()
    rel_list=np.argsort(answ)[::-1] # sorting indexes by a values of sum of ranking vectors
    rel_list=list(map(int,rel_list)) # getting items according to a indexes
    res_list = [pr_items[i] for i in rel_list if pr_items[i] not in S ][:NN] # saving the same number of items as in GT
    res_list=set(map(int,res_list))
    return res_list
#*******************************************

# Recomendations

recom_dict=defaultdict(set)
for u in user_item_dict.keys():
	recom_dict[u]=calc_recom1(user_item_dict[u],u)

#-------------------------------------------
# R - precision
def r_precision(ser,dq0):
    r_p=[]
    val_sum=0
    for k in ser.keys():
        tq=set(list(ser[k])[:len(dq0[k])])
        for x in tq:
            if x in dq0[k]:
                val_sum+=1
        r_p.append(val_sum/len(dq0[k]))
        val_sum=0
    return pd.DataFrame(r_p, columns=['r_p'])


r_p=r_precision(recom_dict,GT)
print("avg of R-precision")
print(r_p.r_p.mean())
print(datetime.datetime.now()-start)
