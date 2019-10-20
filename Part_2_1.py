# =============================================================================
# PART 2.1
# =============================================================================
import os
import datetime
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict

start=datetime.datetime.now()
print(start)

'''
    In order to execute
    put a directory in the command line to execute this file : 
        ./DMT4BaS_2019/HW_2
    then type the following:
        python Part_2_1/sw/Part_2_1.py
'''

path1=os.getcwd()#current working directory

given_graph=nx.Graph() #a bipartite graph (User-Item interactions)
given_graph_edges=[] 
'''
Steps:
    1) Creating the bipartite graph (User-Item interactions)
        * Reading the edges of the bipartite graph from the file
        * Adding nodes of users, nodes of items and edges between them (read from file)

    2) First, we made Projected-Item-Item-Graph from the previously made bipartite graph (User-Item interactions)
    using generic_weighted_projected_graph  method from the NetworkX  library.
    3) For each user we collected the items that are connected to the user as set S (topic), and calculated 
    the rank vector as an output ftom the implemented Topic-Sensitive-PageRank algorithm.
    4) Implementation of the Topic-Sensitive-PageRank algorithm
        • Take adjacency matrix with weights from the Projected-Item-Item-Graph and multiply it with damping factor
        • Items that are considered as part of the topic set S are having probability 1/|S| and other items 0 (first values)
        • Next values for the ranking vector are calculated by the formulas:
           Let's say damping factor is alpha, Pk[x] is the ranking vector.
     For the items that are in the topic values are:
           Pk[x] =alpha*Pk[x] +(1-alpha)/|S|
	 But if the items are not in the topic it is calculated:
           Pk[x] =alpha*Pk[x] +(1-alpha)/|S|
        • This calculation is repeated either until values reach stability (until Euclid distance between 2 last PR vectors isn’t less than or equal to 0.0001 or finish it in max 100 iteration (num_of_iters=100)
        • When Topic-Sensitive-PageRank is calculated for the user,  a recommendation for the user is made
     5) Checking the quaility of recommendation/prediction method by calculating R-Precision
        * 'Ground_Truth___UserID__ItemID.tsv' - Ground truth document used for calculation 
        In this case mean value of R-Precision for all the users
    
'''
#---------------------------------------------
#reading the edges of the bipartite graph from file (User-Item interactions)

with open(path1+"/Part_2_1/dataset/User_Item_BIPARTITE_GRAPH___UserID__ItemID.tsv") as f:
	for line in f:
		given_graph_edges.append(tuple(map(int, line.rstrip('\n').split('\t'))))


#--------------------------------------------
# getting users and items sets separately from the list of edges
users=set()
items=set()
for (u,v) in given_graph_edges:
	users.add(u)
	items.add(v)

#----------------------------------------
# creating a bipartite graph (node attribute named “bipartite” with values 0 or 1 is to identify the sets each node belongs to)
given_graph.add_nodes_from(list(users),bipartite=0) #set of 'users' nodes
given_graph.add_nodes_from(list(items),bipartite=1) #set of 'items' nodes
given_graph.add_edges_from(given_graph_edges)

nx.is_bipartite(given_graph)

#-------------- Projected-Item-Item-Graph -------------------
Prog_graph=bipartite.generic_weighted_projected_graph(given_graph, items)

#----------------------------- Ground truth -------------

GT=defaultdict(list) #dictionary, key=User_id, value=list of items recommended to that user
g_t=list() #list of tuples, (user_id,item_id)
with open(path1+"/Part_2_1/dataset/Ground_Truth___UserID__ItemID.tsv") as f:
	for line in f:
		g_t.append(tuple(map(int, line.rstrip('\n').split('\t'))))

for u, i in g_t: #user_id, item_id in (user_id,item_id)
	temp=set()
	if u in GT.keys():
		temp=set(GT[u]) #set of items for user u in ground truth
		temp.add(i)
		GT[u]=temp
	else:
		GT[u]=set([i])



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Adjacency matrix
A= nx.adjacency_matrix(Prog_graph)#, weight=1)
# in order to get the adjacency matrix of our graph we used function from the library, 
# and took the weight as 1 for all edges according to a formula of TSPR
# and then normalized it by dividing each row of the matrix to the sum of ones in each row

NN=A.get_shape()[0]

alpha=0.1 # damping factor
A_hat=np.zeros([NN,NN])
A_hat=A_hat.astype(float)
items_1=list(items)


A=A.todense()/A.todense().sum(axis=0) #each column should be summed up to 1; axis=0 sum by columns

A=A*0.1#multiplying adjacency matrix with damping factor
# deal with dangling nodes if exist
p=np.zeros(NN)
for x in range(0,NN):
	w=np.sum(A_hat[:,x])
	if w==0:
		p[x]=1

e=np.ones(NN).T
A_hat=A+(e*p.T)/NN

# ***************** TSPR ******************
# in the function below we performed calculation of vector of ranking
# we got one user's items as input

def calc_r(S):
	'''
	Method that calculates page ranks (vector of Topic Specific Page Ranks)
	input: 
		* S - set of topics(items)
	output: page rank vector 
	'''
	# iterative way to obtain pagerank vector
	s_len=len(S) #length of the in

	S_idx=[] #list of indices which are in the set of topics (input S)
	for i in range(0,NN):
		if items_1[i] in S: #if item is in set S(set of specific topics(items))
			S_idx.append(int(i)) #add his index to the list
	P_k=np.zeros((NN,1))
	# as first value for ranking vector we put 1/s_len value in positions of items which were in input set


	for x in S_idx:
		P_k[x]=1/s_len # nodes from the set of items(S) should have equal probability
	num_of_iters=100

	GG=A_hat
	#according to the formula we calculated new adj matrix taking into consideration is i-th item in input set or not


	#starting iterations
	for i in range(0, num_of_iters):
		P_0=P_k
		#in this cycle we are executing multiplication of matrixes

		P_k=np.matmul(GG,P_k)*alpha
		for x in S_idx:
			P_k[x]=P_k[x]+(1-alpha)/s_len
		if distance.euclidean(P_k, P_0) <= 0.0001:
			break
	return (P_k)
#-----------------------------------------------------

# preparing a data for using as topics
user_item_dict=defaultdict(list)#dict, key=
for (u,v) in given_graph_edges:
	if u in user_item_dict.keys():
		temp=user_item_dict[u]
		temp.add(v)
		user_item_dict[u].update(temp)
	else:
		user_item_dict[u]={v}

#-------------------------------------------------------

recom_dict=defaultdict(list)#recommendation dictionary, key=user_id,value=list of recommended items

for x in user_item_dict.keys(): #for each user in the user_item_dict(key=user_id, value=list of items)
	if x in GT.keys():
		nn=len(GT[x])
        #“topic” is all items connected to the user in the original bipartite-graph
		result=calc_r(user_item_dict[x]) #for each user calculate TSPR
		#print(x)
		#print(user_item_dict[x])
		#print(np.argsort(result)[::-1][:10])
		rr=[]
		for i in result.tolist():
			rr.append(i[0])
		#print(sum(rr))
		rel_list=np.argsort(rr)[::-1]#[:nn] #take top nn sorted indices of the result items
		res_list = [items_1[i] for i in list(rel_list) if items_1[i] not in user_item_dict[x]] #take items based on the previous chosen sorted indices
		recom_dict[x]=res_list[:nn]# RECOMMEND NEW items that aren't already connected to this specific ser
		#print(recom_dict[x])
		


#------------------------------------------------------
# R - precision

def r_precision(ser,dq0):
    '''
    Method that calculates R-precision evaluation measure for the quality of Search Engine
    input:
        ser - recommendation dictionary where key=User_id, value=list of items recommended to that user
        dq0 - ground truth, dictionary, key=User_id, value=list of items recommended to that user(TRUE value)
    output:
        R-precision values
    '''
    r_p=[] #list of r_precision values
    val_sum=0 # the counter
    for k in ser.keys():#each user
        tq=set(list(ser[k]))#[:len(dq0[k])])#list of Relevant items for that user
        for x in tq: #for each item id from Relevant_items for that recommendation (user)
            if x in dq0[k]: #if it is in the Relevant_items
                val_sum+=1  # count how many of them is RELEVANT item
        r_p.append(val_sum/len(dq0[k])) #divide by the length of the RELEVANT items from the Ground truth
        val_sum=0#restart the counter
    return pd.DataFrame(r_p, columns=['r_p'])
        

# -------------------- Calculation of R-precision -----------------
r_p=r_precision(recom_dict,GT) #vector of R-precision values for all the users
print("avg of R-precision")
print(r_p.r_p.mean()) #average value of R-precision for all the users
#print(r_p.head())
end=datetime.datetime.now()
print(end-start)


# plotting distribution of R-precision
plt.hist(r_p['r_p'], color = 'blue', edgecolor = 'black')#,density=True
plt.show()