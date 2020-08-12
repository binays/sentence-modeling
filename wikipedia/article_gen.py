import wikipedia
import nltk.data 
import re
from nltk.tokenize import word_tokenize
import json

tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

pos_replace=["CD","NN","NNS","NNP","NNPS","PDT","PRP","PRP$"]
article_id=0
data_count=0

def get_page_content(page_name):
	try:
		return [page_name,wikipedia.page(page_name).content]
	except wikipedia.exceptions.DisambiguationError as e:
		return [e.options[0],wikipedia.page(e.options[0]).content]

def prepare_data_point(con):
	global data_count
	content=con[1]
	title=con[0]
	sentences=tokenizer.tokenize(content)
	#print(sentences)
	pre_final_sent=[]
	for s in sentences:
		if '== See also ==' not in s and '== External links ==' not in s and '== References ==' not in s:
			pre_final_sent.append(s)
	final_sent=[]
	for s in pre_final_sent:
		s=s.replace("\n","")
		s=re.sub("== [a-zA-Z0-9\s]+ ==","",s)
		final_sent.append(s)
	#print(final_sent)
	#Now we have list of sentences in final_sent from each articles with == <text> == and \n removed
	sent_with_pos=[]
	#now for each sentence pair up with its parts of speech
	for s in final_sent:
		wlist=nltk.word_tokenize(s)
		tagged=nltk.pos_tag(wlist)

		pos_list=""
		answer=[]
		for token in tagged:
			pos_list+=token[1]+" "
		for token in tagged:
			word=token[0]
			pos=token[1]
			if pos in pos_replace:
				answer.append((word,pos))

		sent_with_pos.append((article_id,title,s,pos_list,answer))
		d={
		"art_id":article_id,
		"title":title,
		"text":s,
		"pos":pos_list,
		"answer":answer
		}
		with open("/srv/binay/wikipedia/article_"+str(article_id)+".json","a") as outfile:
			json.dump(d,outfile)
		data_count+=1
	#print(sent_with_pos)
	#Now we have data in this format [(article_id,article_title,each_sentence,pos)]







def get_random_pages_content(pages=0):
	global article_id
	page_names=[wikipedia.random(1) for i in range(pages)]
	for p in page_names:
		article_id+=1
		print("Writing article "+str(article_id))
		con=get_page_content(p)
		print(con[0])
		
		prepare_data_point(con)
		f=open("/srv/binay/wikipedia/articles/"+con[0],"w+")
		f.write(con[1])
		f.close()
	return pages


p=get_random_pages_content(pages=100)
print(str(p)+" files written")
print(str(data_count)+" data written")

