import os
import datetime
import pandas as pd
import re
import pickle
print(datetime.datetime.now())
# put a dir in cmd to execute this file : ...\HW_1
# python part_2/sw/part2.py
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

path1 = os.getcwd()
#-------reading a dataset----
dataset = pd.read_csv(path1 + "/part_2/dataset/261K_lyrics_from_MetroLyrics.csv", usecols=['ID', 'lyrics'])

#--- removing punctuation and converting words to lowercase letters---------
dataset['text'] = pd.Series(re.sub(r'[^\w\s]', '', x.lower()) for x in dataset['lyrics'])

# ---- saving only ID and text columns
dataset = dataset[['ID', 'text']]

# method for shingling and mapping all shingles to numbers


def my_shing(lyric):
    list_of_shing = set()  # the set where numbers saved
    num_code = 0
    for x in range(len(lyric)):
        val = ' '.join(lyric[x:x + 3])  # taking one shingle
        for v in val:
            num_code = num_code * 10 + ord(v)
        # with statement above we get some big number from the string
        # we need to map this number to smaller number
        # in order to do that we will use hashing method below 'my_hash'
        num_code = my_hash(num_code)
        list_of_shing.add(num_code)  # adding a hashed number to set of numbers
        num_code = 0  # updating a value for the next iteration
        if x == (len(lyric) - 3):  # stop the loop when it comes to last shing with length 3
            break
    return list_of_shing  # returning a set of numbers for one song

# we picked p from prime numbers
# and we will use it to map big numbers to smaller numbers
p = 165961
def my_hash(num_c):
    '''
    Method for reducing hashed value from the shingled lyric to a smaller one
    '''
    num_b = num_c
    while num_b > 9999999:  # checking if number is more than specified limit

        num_b = num_b // p + num_b % p  # if it is bigger than the limit reduce it

    return num_b

# defining dictionary
Shing_dict = {'A': ['AA', 'AA']}  # dictionary, where key=document_id(song id), values=list of hashed shingles

# counter to see how many song lyrics have less than 3 words in it
too_small_l = 0
list_ids = list(dataset.ID.values)
print(len(list_ids))
#---- filling a dictionary with ids of songs as keys and their sets of numbers that corresponds to shingles as values ----
for x in list(dataset.ID.values):
    if len(dataset[dataset['ID'] == x]['text'].values[0].split()) <= 2:
        too_small_l += 1
        list_ids.remove(x)
print(len(list_ids))
for x in list_ids:
    Shing_dict['id_' + str(x)] = list(my_shing(dataset[dataset['ID'] == x]['text'].values[0].split()))
    if x % 10000 == 0:
        print("10000")
print("too small lyric:  " + str(too_small_l))
print("Shingled lyrics:  " + str(len(Shing_dict.keys())))
del Shing_dict['A']  # deleting not useful key, that we used just for defining a dictionary

# -------- we wanted to see avg of number of shingles in songs
n = 0
for i in Shing_dict.keys():
    # if len(Shing_dict[i])<=3:
        # print(i)
    n = n + len(Shing_dict[i])
n = n / len(Shing_dict.keys())
print("the avg is: " + str(n))
# ----- the method below used for saving dictionary as pkl object in order to use it in other .py file-------------


def save_obj(obj, name):
    with open(path1 + "/part_2/" + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


#-------- preparing tsv file for java tool---------------
Sh_data = pd.DataFrame()
Sh_data['ID'] = list(Shing_dict.keys())
Sh_data['ELEMENTS_IDS'] = list(Shing_dict.values())
Sh_data.to_csv(path1 + '/part_2/dict.tsv', sep='\t', index=False)
# -------------- saving pkl file --------------
save_obj(Shing_dict, "ourdict")
print(datetime.datetime.now())

# ----------------------------------------------------------
# before using java tool
#  we calulated avg of number of shingles in songs, it is n, n is around 169
# so for values r and b we need to pick then in a way when their product is equal to n
# for this purpose we decided to take n as 160 and r = 8 and b=20
# and for using java tool with this value we need to use the hash_functions_creator.py
# in cmd :  python part_2/sw/hash_functions_creator.py
# ***************************************************
# to use that java tool:
# in cmd:
# change a dir to ...\HW_1\part_2
# type :
# java -Xmx3G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.0000 8 20
#./hash_functions/160.tsv dict.tsv
#./data/APPX_0804.tsv

# we picked as threshold as 0 because we need to see cases when JS is 0 also to count FN and FP

# and then to calulate FN and FP probabilities
# execute check_part2.py file:
# in cmd: python part_2/sw/check_part2.py
