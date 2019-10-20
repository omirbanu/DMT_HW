# =============================================================================
# PART 1 - Recommendation-System
#           SVD optimization 
# =============================================================================
'''
Part 1.2

Instructions for running the file in the cmd:
    Change working directory to:
        ./DMT4BaS_2019/HW_2
    run in the command line:
        python part_1/sw/part1_2_svd.py
        
In this scrypt we performed 2 Grid Search Cross Validations over 5 folds to try 
to find the best hyper parameters.
Since execution for first alg were slower we decided to choose parameters wiser.

In first GridSearchCV execution we choose init_mean, lr_all, reg_all
'''

from surprise.model_selection import GridSearchCV

#Matrix Factorization-based algorithm
from surprise import SVD

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

import os
import datetime
'''
Since in the first part 
'''
# Reading the data
file_path = os.path.expanduser('./part_1/dataset/ratings.csv')

print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)#skip header
data = Dataset.load_from_file(file_path, reader=reader)



'''
Optimizing hyperparameters of the SVD
'''
kf = KFold(n_splits=5, random_state=0) #making folds for cross validation


start=datetime.datetime.now()
print('Optimizing hyperparameters of the SVD')
print("Start.....")
print(start)

 
#OPTIMIZATION OF SVD

param_grid = {'init_mean':[0.1,0.15],
              'lr_all':[0.005,0.01,0.025], #0.025 default
              'reg_all':[0.02,0.005,0.1]} #0.1 default
grid_search = GridSearchCV(SVD,param_grid,measures=['rmse'],
                           cv=5,n_jobs=3)
grid_search.fit(data)

  
end=datetime.datetime.now()
print(end-start,"\nEnd.....")
print(grid_search.best_params['rmse'])

#Execution time 0:06:17.595912 
# After first grid search  --> {'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}

'Best found paramteres for SVD'
# =============================================================================
# 0.8838
#{'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}
opt_svd_alg=SVD(init_mean=0.15,lr_all=0.025,reg_all=0.1)
cross_validate(opt_svd_alg,data,measures=['rmse'],cv=kf,n_jobs=3,verbose=True)
# =============================================================================

# =============================================================================
# Try to optimize number of factors by using optimized fixed values for 
# hyper parameter reg_all, lr_all and init_mean
#--> {'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}
# =============================================================================

 
'''
Additional optimization of just number of factors with other chosen fixed hyperparameters
'''

start=datetime.datetime.now()
print('Optimizing hyperparameters of the SVD')
print("Start.....")
print(start)
current_algo = SVD

#OPTIMIZATION OF SVD

param_grid = {'n_factors': [50,100,125,150,200],
              'init_mean':[0.15],
              'lr_all':[0.025],
              'reg_all':[0.1]}
grid_search = GridSearchCV(SVD,param_grid,measures=['rmse'],
                           cv=5,n_jobs=3)
grid_search.fit(data)

#0:02:55.428739 s
  
end=datetime.datetime.now()
print(end-start,"\nEnd.....")
print(grid_search.best_params['rmse'])

# =============================================================================
# 0.8835
opt_svd_alg=SVD(n_factors=150,lr_all=0.025,reg_all=0.1,init_mean=0.15)
cross_validate(opt_svd_alg,data,measures=['rmse'],cv=kf,n_jobs=3,verbose=True)
# =============================================================================
#{'n_factors': 150, 'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}
 
'''
Execution time(I GridSearch+ II GridSearch) ---> 9m13s
'''
##0:06:17.595912 + #0:02:55.428739 =9m13s
#{'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}#(n_factors=150}

 
