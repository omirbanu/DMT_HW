# =============================================================================
# PART 1 - Recommendation-System
#           KNNBaseline optimization 
# =============================================================================
'''
Part 1.2

Instructions for running the file in the cmd:
    Change working directory to:
        ./DMT4BaS_2019/HW_2
    run in the command line:
        python part_1/sw/part1_2.py
'''

from surprise.model_selection import RandomizedSearchCV

#k-NN inspired algorithms
from surprise import KNNBaseline

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

import os
import numpy as np
import datetime



# Reading the data
file_path = os.path.expanduser('./part_1/dataset/ratings.csv')

print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)#skip header
data = Dataset.load_from_file(file_path, reader=reader)


kf = KFold(n_splits=5, random_state=0)

start=datetime.datetime.now()
print('Optimizing hyperparameters of the KNNBaseline')
print("Start.....")
print(start)

'''
Optimizing hyperparameters of the KNNBaseline

In this case we used RandomSearchCV with 30 iterations;
For KNN algorithm the most important parameters, the one that have the biggest 
impact are value of k, which is recommended to be odd value, and simularity 
function
'''
'''
At first we use these parameters just to see if recommendation to use 
pearson_baseline as a similarity function is valid.


similarity_options={
        'name':['cosine','msd','pearson','pearson_baseline'],
        'user_based': [True,False],
        'shrinkage':[1,10,250,100,500,1000,1240]+list(range(50,150))
}
parameters_distributions = {
   
'k': np.arange(1,60,2),
              'min_k':[1,2,3,4,5,6,7,8,9,10,11],
              'sim_options':similarity_options}
'''
#And these were the results
#0.8865
#First iteration 
#{'k': 45, 'min_k': 11, 'sim_options': {'name': 'pearson_baseline', 'user_based': False}}

# =============================================================================
# 'Best found parameters for KNNBaseline in the first Iteration
alg=KNNBaseline(k= 45, min_k=11, sim_options= {'name':'pearson_baseline', 'user_based':False})
cross_validate(alg, data, measures=['RMSE'], cv=kf, verbose=True)
#54m18s was the TIME FOR EXECUTION 
#--->0.8865


'''
Then we tries just the pearson_baseline as a similarity function...
and tried to see the k value which is the best one.
'''


current_algo= KNNBaseline

similarity_options={
        'name':['pearson_baseline'], #it is recommended to use Pearson Baseline
        'user_based': [True,False]
        }
parameters_distributions = {
   
'k': np.arange(1,60,2),
              'min_k':[1,2,3,4,5,6,7,8,9,10,11],
              'sim_options':similarity_options}
searchCV = RandomizedSearchCV(current_algo,
							parameters_distributions,
							n_iter=30,
							measures=['rmse'],
                            n_jobs=3,
							cv=5)
searchCV.fit(data)
end=datetime.datetime.now()
print(end-start,"\nEnd.....")
print(searchCV.best_params['rmse'])

#Second iteration 
#0.8864
#{'k': 37, 'min_k': 11, 'sim_options': {'name': 'pearson_baseline', 'user_based': False}}
alg=KNNBaseline(k= 37, min_k=11, sim_options= {'name':'pearson_baseline', 'user_based':False})
cross_validate(alg, data, measures=['RMSE'], cv=kf, verbose=True)
#22m23s


# =============================================================================
# 'Best found parameters for KNNBaseline 
alg=KNNBaseline(k= 37, min_k=11, sim_options= {'name':'pearson_baseline', 'user_based':False})
cross_validate(alg, data, measures=['RMSE'], cv=kf, verbose=True)
#22m23s --->0.8864
# =============================================================================

