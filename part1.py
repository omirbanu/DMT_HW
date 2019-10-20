# =============================================================================
# PART 1 - Recommendation-System
# PART 1.1
# =============================================================================
'''
Part 1.1

Instructions for running the file in the cmd:
    Change working directory to:
        ./DMT4BaS_2019/HW_2
    run in the command line:
        python part_1/sw/part1.py
        
        
Goal of this part of the task is to try out all the algorithms from the surprise library.
'''

'''
LIBRARIES
'''
#Matrix Factorization-based algorithms
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
#SlopeOne-collaborative filtering algorithm
from surprise import SlopeOne
#k-NN inspired algorithms
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import KNNWithMeans

#CoClustering - collaborative filtering algorithm based on co-clustering
from surprise import CoClustering

#Basic algorithms
#Algorithm predicting the baseline estimate for given user and item.
from surprise import BaselineOnly
#Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal.
from surprise import NormalPredictor

from surprise import Reader
from surprise import Dataset


from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

from tabulate import tabulate
import time
import datetime
import numpy as np
import os
########################################
alg_list=[SVD,SVDpp,NMF,SlopeOne,KNNBasic,KNNWithMeans,KNNBaseline,CoClustering,BaselineOnly,NormalPredictor]
alg_names_lst=['SVD','SVDpp','NMF','SlopeOne','KNNBasic','KNNWithMeans','KNNBaseline','CoClustering','BaselineOnly','NormalPredictor']

# path of dataset file
file_path = os.path.expanduser('./part_1/dataset/ratings.csv')

print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Done.")

print("Performing splits...")
kf = KFold(n_splits=5, random_state=0)
print("Done.")



'''
Print a table of mean RMSE for all the algs
'''
table = []
for idx,klass in enumerate(alg_list):
    print(alg_names_lst[idx],klass())
    start = time.time()
    out = cross_validate(klass(), data, ['rmse'], kf,n_jobs=3,verbose=True)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
    new_line = [alg_names_lst[idx], mean_rmse, cv_time]
    table.append(new_line)
    print('Finished.')
header = ['RMSE','Time']
print(tabulate(table, header, tablefmt="pipe"))
