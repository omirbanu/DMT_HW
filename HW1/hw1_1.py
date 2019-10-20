# =============================================================================
# LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import os
import pylab 
from matplotlib import colors as mcolors
import random

"""
Important memo: 
    * set working directory .\DMT4BaS_2019\HW_1\
    * Order of the execution of the files:
        1. execute hw1.py
        2. execute hw1_1.py
    * In command line
        > python part_1/sw/hw1.py
        > python part_1/sw/hw1_1.py
"""

def read_mrr(path):
    """
    method that returns only the list of indices of the acceptable SE configurations
    input: path to the saved mrrs.csv file(MRR table where first column is SE configuration(text analyzer+scoring function) and second column is mrr value)
    output: list of indices in the MRR table file of acceptable SE configurations
    """
    mrrs=pd.read_csv(path+"mrrs.csv") # reading "mrrs.csv" file
    idx=mrrs[mrrs['1']>0.32].index + 1  #slicing dataframe and taking only rows where mrr value > 0.32; +1 because of the header in csv file
    return list(idx)

def read_mrr_names(path):
    """
    method that returns the list of names of the acceptable SE configurations
    input: path to the saved mrrs.csv file(MRR table where first column is SE configuration(text analyzer+scoring function) and second column is mrr value)
    output: list of names of the acceptable SE configurations in MRR table
    """
    mrrs=pd.read_csv(path+"mrrs.csv") # reading MRR table
    return list(mrrs[mrrs['1']>0.32]['0']) 

path=os.getcwd()+"/part_1/"
list_of_ser=read_mrr(path) #list of indices of the acceptable SE configurations in MRR table
list_of_names=read_mrr_names(path) #list of names of the acceptable SE configurations in MRR table

def create_dict_of_ser(path,ser):
    """
    method that returns dictionary with list of doc_ids in respect to the query id from the results of the SE
    input: path - path to the part_1 folder of the project
           ser - result of the search engine
    output: dd - dictionary, key=Query_id, value=list of doc_ids in respect to the query id
    """
    ser_d=pd.read_csv(path+str(ser)+"__.csv")
    Q=list(ser_d['Query_id'].unique())
    dd={1:list(ser_d[ser_d['Query_id']==1]['Doc_ID'])}
    for i in Q:
        dd[i]=list(ser_d[ser_d['Query_id']==i]['Doc_ID'])
    return dd


gt=pd.read_csv(os.getcwd()+"/part_1/Cranfield_DATASET/cran_Ground_Truth.tsv", sep='\t')
Q=list(gt['Query_id'].unique())
dq={1:list(gt[gt['Query_id']==1]['Relevant_Doc_id'])}
for i in Q:
    dq[i]=list(gt[gt['Query_id']==i]['Relevant_Doc_id']) #dq, dictionary, where key=query_id, value=list of Relevant_Doc_ids

ser1=create_dict_of_ser(path,list_of_ser[0])

def r_precision(ser,dq0):
    '''
    Method that calculates R-Precision
    '''
    r_p=[]
    val_sum=0
    for k in ser.keys(): #ser.keys() are Query_ids
        tq=ser[k] #list of Relevant_Doc_ids for that Query_id
        for x in tq: #for each doc id from Relevant_Doc_ids for that Query_id
            if x in dq0[k]: #if it is in the Relevant_Doc_ids
                val_sum+=1 # count how many of them is RELEVANT docs
        r_p.append(val_sum/len(dq0[k]))  #divide by the length of the RELEVANT docs from the Ground truth
        val_sum=0
    return pd.DataFrame(r_p, columns=['r_p'])


final=pd.DataFrame()
all_dicts=[]
for i in list_of_ser:
	ser1=create_dict_of_ser(path,i)
	r_p=r_precision(ser1,dq)
	nums=[i,r_p.r_p.mean(),r_p.r_p.min(),r_p.r_p.quantile(0.25),r_p.r_p.median(),r_p.r_p.quantile(0.75),r_p.r_p.max()]
	#print(r_p.mean)
	#nums=[i,np.mean(r_p),np.min(r_p),np.percentile(r_p,25),np.median(r_p),np.percentile(r_p,75),np.max(r_p)]
	nums=pd.DataFrame(nums).T
	final=final.append(nums)
	all_dicts.append(ser1)
    
#Making the R-Precision distribution table    
final.columns=['ser_id','mean','min','1quartile','median','3quartile','max']
final=final.reset_index()
final['SE conf']=list_of_names
final=final[['ser_id','SE conf','mean','min','1quartile','median','3quartile','max']]
print(final)
final=final.round(3)
final.to_csv('part_1/R-Precision distribution table.csv')


def myDCG(query_r,query_gt,k):
	s=0
	dsg=[]
	for x in range(len(query_r[0:k+1])):
		if query_r[x] in query_gt[0:k+1]:
			s=1
		else:
			s=0
		#print(str(s)+" "+str(x)+" "+str(s/np.log2(x+1)))
		dsg.append(s/np.log2(x+2)) # we added +1 to formula from lab because counting starts from 0 in python
		#dsg=dsg+s
	return sum(dsg)

# =============================================================================
# Calculation of nDCG: normalized Discounted Cumulative Gain
# =============================================================================

def myDCG_for_ser(ser_list,all_dict,dq,k):
	s_d=pd.DataFrame()
	for i in range(len(ser_list)):
		dcg=[]
		w_dict=all_dict[i]
		for y in w_dict.keys():
			dcg.append(myDCG(w_dict[y],dq[y],k))
		s_d_temp=pd.DataFrame(dcg).T
		s_d=s_d.append(s_d_temp)
	return s_d

#AA=myDCG_for_ser(list_of_ser,all_dicts,dq,2)

def n_myDCG(DCG):
	for i in range(DCG.shape[1]):
		if (np.max(DCG[i])>0):
			DCG[i]=DCG[i]/np.max(DCG[i])
	return DCG

def avg_s_dcg(DCG):
	avg_of_k=[]
	for i in range(DCG.shape[0]):
		#print(list(list(DCG[i:(i+1)].values)[0]))
		avg_of_k.append(np.mean(list(list(DCG[i:(i+1)].values)[0])))
	return avg_of_k

#creating DataFrame with nDCG values
k_nDCG=pd.DataFrame()
for k in range(1,len(list_of_names)):
	DCG=myDCG_for_ser(list_of_ser,all_dicts,dq,k)
	nDCG=n_myDCG(DCG)
	avgs=avg_s_dcg(nDCG)
	k_nDCG[k]=avgs
k_nDCG=k_nDCG.T
print(k_nDCG)

# =============================================================================
# nDCG@k plot for accepted configurations
# =============================================================================

def nDCG_plot():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) #list of colors
    cols=random.sample(list(colors.keys()),len(list_of_names)) #take random color sample from colors list->length of sample is number of accepted configurations

    pylab.figure(figsize=(10,10))    
    for i in range(0,len(list_of_names)):
        y = k_nDCG[i]#nDCG@k values for the accepted configurations
        pylab.plot(k_nDCG.index, y,label=list_of_names[i],color=cols[i])
        pylab.legend(loc='upper left',prop={'size': 10})

    pylab.savefig('part_1/nDCG@k.png')

    pylab.show()
nDCG_plot()