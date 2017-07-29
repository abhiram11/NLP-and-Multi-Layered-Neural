import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # similar to stemming 
import numpy as np 
import random #toshuffle data 
import pickle # to ssave data
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000
#memory error is possible according to number of lines, running out of RAM/V-RAM

def create_lexicon(pos,neg):
	lexicon = [] #initially empty
	for fi in [pos,neg]:
		with open(fi,'r', encoding='cp437') as f: #encoding parameter added for python 3.6
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon+=list(all_words)


	#populated lexicon with every word encountered
#removing extra words
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	#legitimate words found
	w_counts = Counter(lexicon)

	l2 =[]
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)
			print(len(l2)) #every input vector will have 423 elements to it, i.e lexicon value
	return l2 #final lexicon
	#because this will remove the SUPER COMMON Words

#now use lexicon to classify feature sets



def sample_handling(sample, lexicon, classification):
	featureset = []
	with open(sample,'r', encoding='cp437') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon)) # features now has an empty array of length same as the lexicon
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] +=1
				features = list(features)
				featureset.append([features, classification])
	return featureset

#conclusion functions
def create_feature_sets_and_labels(pos,neg,test_size=0.1):
	lexicon = create_lexicon(pos,neg)
	features=[]
	features += sample_handling('pos.txt',lexicon,[1,0]) #the return will be sent here
	features += sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(features) #shuffles positive and negatives

	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])  #list of all the zeroth elements of multiple arrays [[5,8],[9,7],[2,3]] = [5,9,2]
	train_y = list(features[:,1][:-testing_size]) #down to the last elememts of testing size

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

if __name__ == '__main__':

	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	with open('abhiram_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)





