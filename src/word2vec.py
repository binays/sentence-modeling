import gensim


file=open("quora_duplicate.tsv","r",encoding="UTF8")
text=open("text.txt","a")
label=open("labels.txt","a")
for line in file:
	error=0
	try:
		linearr=line.split("\t")
		x=linearr[3]+" "+linearr[4]
		y=linearr[5]
		
	except:
		print("error")
		error=1
	if error==0:
		text.write(x+"\n")
		label.write(y)

text.close()
label.close()
	
