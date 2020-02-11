import random

text=open("quora_duplicate.tsv","r").readlines()
labels=open("labels.txt","r").readlines()
dup_questions=[]

non_dup_questions=[]

dup_y={}
non_dup_y={}

for i,line in enumerate(text):
	try:
		linearr=line.split("\t")
		q=linearr[3]+"\t"+linearr[4]
		l=linearr[5]
		
		if int(l)==1:
			dup_questions.append(q)
		else:
			non_dup_questions.append(q)
			
	except:
		print(len(linearr))
		print(line)

print("duplicate:"+str(len(dup_questions)))
print("non dup:"+str(len(non_dup_questions)))

no_valid=4000
no_test=5000

dup_valid_question=random.sample(dup_questions,no_valid)
non_dup_valid_question=random.sample(non_dup_questions,no_valid)

valid=open("validation.txt","a")

for q in dup_valid_question:
	valid.write(q+"\t"+str(1)+"\n")

for q in non_dup_valid_question:
	valid.write(q+"\t"+str(0)+"\n")

valid.close()
print("validation done")

dup_questions=list(set(dup_questions)-set(dup_valid_question))
non_dup_questions=list(set(non_dup_questions)-set(non_dup_valid_question))
print("duplicate:"+str(len(dup_questions)))
print("non dup:"+str(len(non_dup_questions)))

dup_test_question=random.sample(dup_questions,no_test)
non_dup_test_question=random.sample(non_dup_questions,no_test)

test=open("test.txt","a")
for q in dup_test_question:
	test.write(q+"\t"+str(1)+"\n")

for q in non_dup_test_question:
	test.write(q+"\t"+str(0)+"\n")

test.close()
dup_questions=list(set(dup_questions)-set(dup_test_question))
non_dup_questions=list(set(non_dup_questions)-set(non_dup_test_question))
print("duplicate:"+str(len(dup_questions)))
print("non dup:"+str(len(non_dup_questions)))

training=open("training.txt","a")
for q in dup_questions:
	training.write(q+"\t"+str(1)+"\n")

for q in non_dup_questions:
	training.write(q+"\t"+str(0)+"\n")

training.close()



