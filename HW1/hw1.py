# =============================================================================
# LIBRARIES
# =============================================================================

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import StemmingAnalyzer,SimpleAnalyzer,StandardAnalyzer,RegexAnalyzer
from whoosh.analysis import FancyAnalyzer,NgramAnalyzer,KeywordAnalyzer,LanguageAnalyzer

from whoosh.writing import AsyncWriter
import os
from whoosh import index
from whoosh.qparser import *
from whoosh import scoring
import csv

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import collections


"""
Important memo: 
    * set working directory .\DMT4BaS_2019\HW_1\
    * In command line type:
        > python part_1/sw/hw1.py
        > python part_1/sw/hw1_1.py
    * Order of the execution of the files:
        1. execute hw1.py
        2. execute hw1_1.py
        
put 'Cranfield_DATASET' folder in part_1 folder
"""

def creating_csv_from_html(path):
    """
    Method that returns a dataframe with columns ['ID','title','body'] from content of 1401 html pages stored in directory
    input: path to the .html files
    output: dataframe ['ID','title','body'] from the content of the .html pages 
    """
    document=pd.DataFrame(columns=['ID','title','body']) #initialization of the dataframe 
    doc_ids=range(1,1401) #number of .html files stored in the folder

    for i in doc_ids: #looping through each of the html files
        filename=path+'_'*6+str(i)+'.html' #storing file name
        with open(filename) as f:
            content = f.read() #reading content of the file

        soup = BeautifulSoup(content, 'html.parser') #taking html content with BeautifulSoup library and parsing 
        title=soup.title.string # storing title from parsed html content
        body=soup.body.string # storing body from parsed html content


    
        x=len(body.split(' ')) #check if there is document without body, with only one character (we found 2 documents with just '.' in the title and without body)
        if x<2:    
            continue #documents with ids :
        document=document.append({'ID':i,'title':title,'body':body},ignore_index=True) #adding new document to the dataframe 
        # each new row is each new content taken from new parsed .html file

    return document

path1= os.getcwd()+"/part_1/Cranfield_DATASET/DOCUMENTS/"#path to the html files
files_text=creating_csv_from_html(path1) #calling the method and storing the dataframe into files_text variable
files_text.to_csv(os.getcwd()+"/part_1/Cranfield_DATASET/csv_test.csv") #saving the dataframe into Cranfield_DATASET folder


def creating_searching_ranking(selected_analyzer, name_of_file,scoring_function,path):
    """
    Method that creates schema and stores index file based on the retrieved 'csv_test.csv' file  
    input:  
        selected_analyzer - selected text analyzer from the whoosh library
        name_of_file - name of .csv file stored from dataframe variable 'files_text'
        scoring_function - selected scoring function from the whoosh library
        path - path where index files are stored
    """
	#creating Schema with fields id, title and content
    schema = Schema(id=ID(stored=True),\
				title=TEXT(stored=False, analyzer=selected_analyzer),
				content=TEXT(stored=False, analyzer=selected_analyzer))
    directory_containing_the_index = path 
    ix = create_in(directory_containing_the_index, schema) #vrating index based on schema in the directory where the 'path' is
    directory_containing_the_index = path
    ix = index.open_dir(directory_containing_the_index) #opening the index file 
    writer =  AsyncWriter(ix) #writer will be used to add content to the fields

	#num_added_records_so_far=0
    ALL_DOCUMENTS_file_name = name_of_file #path to the file 
    in_file = open(ALL_DOCUMENTS_file_name, "r", encoding='latin1')
    csv_reader = csv.reader(in_file, delimiter=',')  #reading the file
    csv_reader.__next__()# to skip the header: first line contains the name of each field.
	#num_added_records_so_far = 0
    for record in csv_reader: #for each row in the 'csv_test' file 
        id = record[1] #read id
        title = record[2] #read title
        content = record[3] #read body
        writer.add_document(id=id, content=title+' '+content)
		#num_added_records_so_far +=1
		#if (num_added_records_so_far%1000 == 0):
		#    print(" num_added_records_so_far= " + str(num_added_records_so_far))

    writer.commit()
    in_file.close() #finish writing in the index file
    

def exec_searching_ranking(selected_analyzer,scoring_function,input_query,path,max_number_of_results):
    '''
    Method that given the input query and given the specific SE configuration returns the results of the search
    input:
        selected_analyzer - selected text analyzer from the whoosh library
        scoring_function - selected scoring function from the whoosh library
        input_query - query that's being used for evaluation
        path - path where index files are stored
        max_number_of_results - maximal number of results that should be retrieved which are equal to the number of relevant documents related to that specific query (which we need for calculating R-precision)
    output: answer - dataframe with results of the given SE given the query; columns of dataframe: ["Rank" , "Doc_ID" , "Score"]
    ''' 
    
    directory_containing_the_index = path 
    ix = index.open_dir(directory_containing_the_index) #index file for the given SE
    qp = QueryParser("content", ix.schema)
    parsed_query = qp.parse(input_query)# parsing the INPUT query
    #print("Input Query : " + input_query)
    #print("Parsed Query: " + str(parsed_query))

    searcher = ix.searcher(weighting=scoring_function) #defining scoring_function for search engine

    results = searcher.search(parsed_query,limit=max_number_of_results) #saving results of the query and limiting max number of results

    #print("Rank" + "\t" + "DocID" + "\t" + "Score")
    answer=pd.DataFrame()  # dataframe with results for the given SE given the query
    row_answer=pd.DataFrame()
    for hit in results:
      #  print(str(hit.rank) + "\t" + hit['id'] + "\t" + str(hit.score))
        row_answer=pd.DataFrame([str(hit.rank) , int(hit['id']), str(hit.score)]).T
        answer=answer.append(row_answer)
    answer.columns=["Rank" , "Doc_ID" , "Score"]
    searcher.close()
    return answer

def exec_queries(selected_analyzer,scoring_function):
    '''
    Method that given the specific SE configuration(selected_analyzer,scoring_function)
    executes and returns the results for ALL the queries 
    input:
        selected_analyzer - selected text analyzer from the whoosh library
        scoring_function - selected scoring function from the whoosh library
    output: answer_q - dataframe with the results of the given SE for ALL the queries; columns of df: ["Rank" , "Doc_ID" , "Score"]
    ''' 
	
    answer_q=pd.DataFrame() #  dataframe with the results of the given SE for ALL the queries; 
    aa=pd.DataFrame() #tmp dataframe 
    
    #all the queries file
    Queries_file=os.getcwd()+"/part_1/Cranfield_DATASET/cran_Queries.tsv"
    Queries=pd.read_csv(Queries_file,sep='\t')
    gt=pd.read_csv(os.getcwd()+"/part_1/Cranfield_DATASET/cran_Ground_Truth.tsv", sep='\t') #ground truth
    Q=list(gt['Query_id'].unique()) #list of unique Query ids
    
    dq=collections.defaultdict(int) #dictionary, where key=Query_id, value=number of relevant documents related to that Query_id
    for i in Q: # for each query)_id
    	dq[i]=len(list(gt[gt['Query_id']==i]['Relevant_Doc_id']))
	
    
    name_of_file_1=os.getcwd()+"/part_1/Cranfield_DATASET/csv_test.csv"
	
    # calling the method that creates schema and stores index file based on the retrieved 'csv_test.csv' file  
    creating_searching_ranking(selected_analyzer,name_of_file_1,scoring_function,os.getcwd()+"/part_1/")
    for i in Q:
		#print(i)
        max_number_of_results_1q=count_of_vals(dq,i)
        if max_number_of_results_1q==0:
            max_number_of_results_1q=1
            
        # calling the method that given the input query and given the specific SE configuration returns the results of the search
        aa=exec_searching_ranking(selected_analyzer,scoring_function,list(Queries[Queries['Query_ID']==i]['Query'])[0],os.getcwd()+"/part_1/",max_number_of_results_1q)
        aa['Query_id']=i
        answer_q=answer_q.append(aa)#[['Query_id',1]] APPEND dataframe for each query
	#answer_q.columns=['Query_id','Doc_ID']
    return answer_q
          
def count_of_vals(dq,q_n):
    '''
    Method that returns number of relevant documents related to the specific input query
    input:  dq - dictionary, where key=Query_id, value=number of relevant documents related to that Query_id
            q_n - Query_id
    output: number of relevant documents
    '''
    return dq[q_n]
#sr_1=exec_queries()
    
def exec_comp():
    '''
    Method that calculates MRR: Mean Reciprocal Rank and saves a table with MRR evaluation for every search engine configuration 
    '''
    #text analyzers
    selected_analyzers = [StemmingAnalyzer(),SimpleAnalyzer(),StandardAnalyzer(),RegexAnalyzer(),FancyAnalyzer(),NgramAnalyzer(5),KeywordAnalyzer(),LanguageAnalyzer('en')]#text analyzers
    sel_ana=['StemmingAnalyzer()','SimpleAnalyzer()','StandardAnalyzer()','RegexAnalyzer()','FancyAnalyzer()','NgramAnalyzer(5)','KeywordAnalyzer()','LanguageAnalyzer()']#text which will be used for graph and for mrr table
	
    i=0 #counter
    mrrs=[] #list where MRR values for each SE configuration will be stored

    #scoring functions
    scoring_functions = [scoring.TF_IDF(),scoring.Frequency(),scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)]
    scor_func=[' TF_IDF',' Frequency',' BM25F']
	
    #ground truth
    gt1=pd.read_csv(os.getcwd()+"/part_1/Cranfield_DATASET/cran_Ground_Truth.tsv", sep='\t')
    
    #combinations for every chosen analyzer with every chosen scoring function
    for x in range(len(selected_analyzers)):
        for y in range(len(scoring_functions)):
            print(sel_ana[x]+scor_func[y])
            i=i+1
            sr_1=exec_queries(selected_analyzers[x],scoring_functions[y]) # execute queries for the chosen configuration combination
            sr_1.to_csv(os.getcwd()+"/part_1/"+str(i)+"__.csv",index=False) #save results of the search engine
            mrrs.append((sel_ana[x]+scor_func[y],mrr(gt1,sr_1))) #calculate MRR
    mrrs_saving=pd.DataFrame(mrrs)
    mrrs_saving.to_csv(os.getcwd()+"/part_1/mrrs.csv", index=False) #store MRR table
	#return [gt1, sr_1]



def mrr(gt,sr1):
    '''
    Method that calculates MRR: Mean Reciprocal Rank
    input: gt - ground truth
           sr1 - search engine results, dataframe with columns (Rank, Doc_ID, Score, Query_id)
    output: mrr value for that specific input 'sr1' (search engine configuration)
    '''
    
    mrr=0
    Q=set(gt['Query_id'].unique()) #number of unique queries in the ground truth
    dd=collections.defaultdict(int) #default dictionary, where (key=Query_id,value=list of document ids) from se result

    for i in Q:
        dd[i]=list(sr1[sr1['Query_id']==i]['Doc_ID'])

    dq=collections.defaultdict(int) #default dictionary, where (key=Query_id,value=list of relevant document ids) from ground truth
    for i in Q:
        dq[i]=list(gt[gt['Query_id']==i]['Relevant_Doc_id'])
   
    tq=list() #->temporary list- stores list of relevant doc_ids for every query id in the loop
	#print(dd[1])
	#print(dq[1])
    for q in Q: #for every query_id
        tq=dq[q]
		
        for i in range(len(dd[q])): #for every document_id in query_id q
			
            #if document id is in the list of the relevant doc ids 
            if dd[q][i] in tq: # dd[q]-list of document ids with query_id q -> [i] is index of list
				#print(i)
                mrr=mrr+(1/(i+1))	#mrr value is sum on Reciprocal Ranks (+1 cause ranking ofc starts with 1)
                break #if it is break cause it found the first doc id from the relevant doc ids in the ground truth

	#print("1 :")
	#print(mrr/Q)
    mrr=mrr/(len(Q)) #MEAN of the sum of reciprocal ranks
    return mrr


exec_comp() #execution of the file with all the methods -> main call
#sr_1=exec_queries(StemmingAnalyzer(),scoring.TF_IDF())

