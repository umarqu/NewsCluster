import getpass,requests,json,requests.auth,os,hashlib,unicodedata
from time import sleep
from lxml import etree
from io import StringIO
from bs4 import BeautifulSoup, Comment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from collections import Counter
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, ClusterMixin

import codecs
import numpy as np

#Login details. the request library
username = "umar_00" #username, only works with mine 
USER_AGENT = "python:newsCluster/0.1 (by /u/umar_00)" # the newCluster string is random
SCR = "vRqXwgCeBNreLOXVyivIfLnauEk" #comes frm the reddit ap config
ID = "gtbMSO5im7e2zg" 
subReddit = "r/worldnews"
password = ""
# create the below dir from your root Projects/NewsCluster/raw
FolderToFiles = os.path.join(os.path.expanduser("~"), "Projects","NewsCluster", "raw")
text_output= os.path.join(os.path.expanduser("~"), "Projects","NewsCluster","Raw_Text")


def collectLogin():
	usr = raw_input("Please enter your username: ")

def login(usr,scr,id,password):	
	pas = getRedditPassword()
	headers = {"User-Agent": USER_AGENT,"Authorization": "bearer TOKEN"}
	client_auth = requests.auth.HTTPBasicAuth(id, scr)
	post_data = {"grant_type": "password", "username": usr,
				"password": pas}
	response = requests.post("https://www.reddit.com/api/v1/access_token", 
			auth=client_auth, data=post_data, headers=headers)
	return response.json()
	
def getRedditPassword():
	password = ''
	password=getpass.getpass("Enter Reddit Password {} :".format(password))
	return password	

def getRedditNews(Token):
	#token is a dic
	#header is used to allow authorization and restritions wont occur
	headers = {"User-Agent": USER_AGENT, 
			   "Authorization": "bearer {}".format(Token['access_token'])}
	response_result = requests.get("https://oauth.reddit.com/r/worldnews", headers=headers)
	return response_result.json()
	
def getLinks(subReddit,Reddit_token):
	Story_List = []
	p_number = 5 # this is fro the iteration
	after = None
	for page_number in range (p_number):
		headers = {"Authorization": "bearer {}".format(Reddit_token['access_token']),"User-Agent": USER_AGENT}
 		URL = "https://oauth.reddit.com/{}?limit=500".format(subReddit)
 		if after: #until none is reached
 				URL +="&after={}".format(after) # creates new url ned
 		response = requests.get(URL, headers=headers)	
 		resultSet = response.json()
		after = resultSet['data']['after']
		#sleep(2) # just in case Api exeeds limits
		for newStories in resultSet['data']['children']:
			# adding the title,score,url of the news only 
			Story_List.extend([newStories['data']['title'],newStories['data']['url'],
									newStories['data']['score']])
	return Story_List

def charParse(List_item):
	for title in range(0,len(List_item),3):
		List_item[title] = ''.join(i for i in List_item[title] if ord(i)<128)
 	return List_item

def downloadHTML(news_Stories):
	errorFilesCounter = 0
	news_Stories = charParse(news_Stories) #remove problem causing chars
	for url in range(0,len(news_Stories),3): #loop from 0 to list end, iterate every 3
		link=news_Stories[url]
		o_fileName =  hashlib.md5(link.encode()).hexdigest() # hashing the title
 		filepath = os.path.join(FolderToFiles, o_fileName + ".html") #file to the path
 		try:
 			response = requests.get(news_Stories[url+1])# +1 every 3rd from 1st index = URL
 			dataFromSite = response.text 
 			F= open(filepath,'w') 
 			#converting unicode->string so we can write
 			#dataFromSite =unicodedata.normalize('NFKD', dataFromSite).encode('ascii','ignore')
 			#if dataFromSite=="Forbidden" or dataFromSite==" ":
 			#	continue 
 			F.write(dataFromSite.encode('utf8'))
 			F.close()
 		except Exception as e:
 				errorFilesCounter = errorFilesCounter+1
 		#print("number of errors",errorFilesCounter)

def urls(List_item):
	for title in range(0,len(List_item),3):
		List_item[title] = ''.join(i for i in List_item[title] if ord(i)<128)
 		print(List_item[title])

def extractRawText():
	#List of File paths to each Raw file in Raw
	Dir_RawFiles= [os.path.join(FolderToFiles, filename) for filename in os.listdir(FolderToFiles)]
	
	#creating a new file top put the new information in
	text_output_folder = text_output

	#extracting text, leaving out all else. Scripts javascript etc
	skipping_node_element=  ["script", "head", "style", etree.Comment]	
	#newDir1 = [Dir_RawFiles[0],Dir_RawFiles[1],Dir_RawFiles[2]]
	helper_textFromFiles1(Dir_RawFiles,text_output_folder)
	#results = (get_text_from_node(child) for child in node if child.tag not in skip_node_types)
	

def helper_textFromFiles1(directory,newDirectory):
	directorySubString = "/raw/" #using to extract filename
	for files in directory:
		f = open(files)
		#creates soup object . f = file
		soup = BeautifulSoup(f, 'html.parser') 
		#remving tags from script we dont need
		soup = removeElementsFromScript(soup)
		#saving just the text minus the tags
		text = soup.get_text()
		#getting the filename , used to save the new file with new text
		newFileName = getFileNamefromDirectory(files,directorySubString)
		#writes just text ti new file
		helper_WriteToFile(text,newDirectory,newFileName) 



def removeElementsFromScript(soupScript):
	#Taking out the tages ,leaving just the text
	comments = soupScript.findAll(text=lambda text:isinstance(text, Comment))
	[comment.extract() for comment in comments]
	[x.extract() for x in soupScript.findAll('script')]
	[x.extract() for x in soupScript.findAll('style')]
	[x.extract() for x in soupScript.findAll('header')]
	[x.extract() for x in soupScript.findAll('footer')]
	#[x.extract() for x in soup.findAll('title')]
	return soupScript

def getFileNamefromDirectory(dir_file,parentDirectory):
	IndexAtFile = dir_file.index(parentDirectory) +5 #index after /raw/
	fileName = dir_file[IndexAtFile:len(dir_file)-5] # new folder string
	return fileName

def helper_WriteToFile(fileText,newDirectory,newFileName):
	#write to new file in new directory
	filepath = newDirectory + "/" + newFileName + ".txt" #file to the path
	if len(fileText) ==0:
	#nothing in the file
		pass
	if len(fileText) < 100:
		pass
	else:
		try:
 			F=open(filepath,"w") 
 			#converting unicode->string so we can write
 			#dataFromSite =unicodedata.normalize('NFKD', dataFromSite).encode('ascii','ignore')
 			#if dataFromSite=="Forbidden" or dataFromSite==" ":
 			#	continue 
 			F.write(fileText.encode('utf8'))
 			F.close()
 		except Exception as e:
 				print(e)

	
def helper_textFromNode(node):
	if len(node) ==0:
	#nothing in the node
		if node.text and len(node.text) > 150:
			return node.text
		else:
			return ""
	results = (helper_textFromNode(child) for child in node if child.tag not in skip_node_types)
	#print ("results",results)
	return "\n".join(r for r in results if len(r) > 1)

def vectorisingDocument():
	#Method should return a list of string representing the text files
	#In this form list[document1,document2,document3, ...documentN] N=total number of files
	Dir_RawFiles= [os.path.join(text_output, filename) for filename in os.listdir(text_output)]
	vectorizer = CountVectorizer(min_df=1)
	document = []
	text = ""
	for files in Dir_RawFiles:
		with codecs.open(files, "r",encoding='utf-8', errors='ignore') as f:
			text = f.read().replace("\n", "")
			document.append(text)
	return document


def KmeansAlgoEnsemble(document):
	#X = vectorizer.fit_transform(document)
	#X = TfidfVectorizer(document,max_df=0.4)
	Number_of_Clusters = 10
	#  Transforms text to feature vectors . max df at 43%,  words appearing
	tfidf_vectorizer = TfidfVectorizer(max_df=0.43, min_df=2,max_features=100)
	#tfidf = tfidf_vectorizer.fit_transform(document)
	pipeline = Pipeline ([('feature_extraction', tfidf_vectorizer),('clusterer', KMeans(n_clusters=Number_of_Clusters))])
	#Pipeline of transforms with a final estimator.
	pipeline.fit(document)
	#  same labels values are said to belong in the same cluster
	label = pipeline.predict(document)
	#ClusterTopWords(pipeline,Number_of_Clusters,label)
	ensembleCluster(label,document,pipeline)
	

def ClusterTopWords(pipeline,Number_of_Clusters,label):
 	words = pipeline.named_steps['feature_extraction'].get_feature_names()
 	# words via the pipeline . feature is key , returns names
 	c = Counter(label) #label for counting the sized of classes
	for cluster_number in range(Number_of_Clusters): #10
		 # prints cluster number and number of samples
 		print("Cluster {} has {} samples".format(cluster_number,c[cluster_number]))
 		print("Top words in Clusters ",cluster_number)
 		#getting largets values from the venter, returns the features (words)

 		centroid = pipeline.named_steps['clusterer'].cluster_centers_[cluster_number]
 		important_words = centroid.argsort()
 		for k in range(6): # printing out the words , top 6
 			wordAtindex = important_words[-(k+1)] #soted from the lowest
 			print("  {0}) {1} (score: {2:.4f})".format(k+1, words[wordAtindex], centroid[wordAtindex]))


def ExtractTopicInformation(document):
	tfidf_vectorizer = TfidfVectorizer(max_df=0.43, min_df=2,max_features=100)
	pipeline = Pipeline ([('feature_extraction', tfidf_vectorizer),('clusterer', KMeans(n_clusters=10))])
	pipeline.fit(document)
	words = pipeline.named_steps['feature_extraction'].get_feature_names()
	print(words)

def OptimumClusterValue(document):
 	# number of clusters is at 10, we are going to find the best number of clusters to use
 	#following part runs k-Means algorithm 10 times, with cluster number increasing 2..20
 	#each run , nerita of result is recorded
 	intertia_values = []
 	n_clusters = list(range(1,20))
 	for clusterNumber in n_clusters:
 		current_inertia = []
 		X =  TfidfVectorizer(max_df=0.4).fit_transform(document)
 		for i in range(10):
 			print(i)
 			K_Means = KMeans(n_clusters = clusterNumber,n_jobs = -2).fit(X)
 			current_inertia.append(K_Means.inertia_)
 		intertia_values.append(current_inertia)
 	print(intertia_values)
 	#should be 7


def co_associate_matrix(labels):
	#AEC algorithm evidence based
	rows = []	#empty rows
	columns = [] #empty column
	uniqueLabels = set(labels)
	for label in uniqueLabels:
		samples = np.where(labels ==label)[0]
		#for each pair of sample with preceding label , save postion
		for index_1 in samples:
			for index_2 in samples:
				rows.append(index_1)
				columns.append(index_2)
	#value for everytime samples were listed together. 

	data = np.ones((len(rows),))
	return csr_matrix((data, (rows, columns)), dtype='float')

def ensembleCluster(label,document,pipeline):
	C_Matrix1 = co_associate_matrix(label)
	#combines multiple runs of k , shows how manu non zeros exist
	#hierar cluster of above matrix via min spanning tree on matrix and removing nodes with lower weight than thresh
	# spanning tree is edges that connects all nodes. min span treeis lowest in weight
	#hence nodes = samples from data, edge weight = number of times samples were clusterd.  
	min_span_tree = minimum_spanning_tree(-C_Matrix1)
	label2 = pipeline.predict(document)
	#adding extra labels 
	C_Matrix2 = co_associate_matrix(label2)
	C_Matrix_Sum = (C_Matrix1+C_Matrix2)/2
	#removing edge that didnt accur 
	min_span_tree = minimum_spanning_tree(-C_Matrix_Sum)
	min_span_tree.data[min_span_tree.data > -1] = 0
	n_clusters, labels3 = connected_components(min_span_tree)
	print("After Esemble")
	ClusterTopWords(pipeline,n_clusters,labels3)



password = getRedditPassword() # asks user once , keeps password
#Login_Token = login(username,SCR,ID,password) # Login via password and returns token
#result_set = getRedditNews(Login_T
#Story_Links = getLinks(subReddit,Login_Token) # title,url and score of 500 news articles
#downloadHTML(Story_Links)
#urls(Story_Links)
#extractRawText()
document = vectorisingDocument()
KmeansAlgoEnsemble(document)
#OptimumClusterValue(document)
#ExtractTopicInformation(document)
	
