# Sentence Modeling

In this project of sentence modeling we study the techniques of forming the representation of the sentences either in a pair or separately . This has many practical applications. Specifically, we started with a discriminative task to be solved by sentence pair modeling called Paraphrase Identification. The task is to model a given pair of sentences and decide whether they are paraphrase of each other. In this work, we have employed various neural architectures to do the task. We experiment with variants of Recurrent Neural Networks and also design a convolutional neural network for the task.

Next, we explore one of the generative tasks in sentence modeling called Question Generation. The task is to form a appropriate question given an input sentence or answer. A closely related task is called Reading Comprehension where given a passage of text and a question, we have to find an answer from that passage. 

There are many variants of Question Generation. One of the recent work is done by Du et al[1]. Given a sentence or a paragraph and a target answer, they try to generate question appropriate for that target answer.

#### Sentence
> Oxygen is used in cellular respiration and released by ***photosynthesis***, which uses the energy of ***sunlight*** to produce oxygen from ***water***.

#### Question for target answer photosynthesis 
> What life process produces oxygen in the presence of light?
#### Question for target answer sunlight
> Photosynthesis uses which energy to form oxygen from water?
#### Question for target answer water
> From what does photosynthesis get oxygen?

In this work, they use a RNN encoder-decoder architecture and a global attention mechanism.


Another work done by Kim et al[2]. is called Neural question generation where they argue the problem with question generation is that the significant portion of generated question contains the words from the target answer. For example:

#### Passage
> ***John Francis O'Hara*** was elected president of Notre Dame in 1934.

#### Improperly generated Question
> Who was elected John Francis?

So they employ the technique of answer masking while encoding the input passage.
> "Masked" was elected president of Notre Dame in 1934.

After masking the answer, they encode the masked passage using a RNN encoder and use the module called keyword-net to leverage the information from target answer. In previous case, "John Francis O'Hara".

Other related works are [3] and [4].

## Our Problem
We want to tackle a slightly different version of question generation. We argue that the model should be able to generate relevant question without even seeing the target answer. We humans are capable of doing that almost all of the time. For example if we hear the following sentence:

> ... was the 16th president of the United States.

In the above sentence, even if we didn't hear the first phrase, we can ask the question "Who was the 16th president of the United States?"

Most of the existing question generation tasks encode the whole sentence "Abraham Lincoln was the 16th president of the United States." to generate the desired question. That means, they extract information from the phrase "Abraham Lincoln" to form the question. But that doesn't seem natural. Humans can ask question without listening to the key answer phrase. Above discussed work of Kim et al.[2] masks the target answer while encoding it but use them separately while forming the question. So through our work, we want to create model that can ask question as if it didn't understand the input sentence.

This work is a step forward in Natural Language Understanding. One practical application of this task can be asking relevant question to your queries by Smart speaker or home assistant like Google Home or Amazon Alexa.

For example when we ask, Hey google, Play ... song. If it doesn't understand the song name you requested, it can ask "What song do you want me to play?" Currently, it says standard sentence like "I didn't understand what you said." or "I'm not sure about that."

We plan to use Transformer architecture for the task.

The dataset we will be using can be downloaded [here](https://www.cs.rochester.edu/~lsong10/downloads/nqg_data.tgz). This dataset is prepared by Kim et al. from the original SQuAD dataset created for Reading Comprehension.
The data format is:
```
[{"text1":"IBM is headquartered in Armonk , NY .", "annotation1": {"toks":"IBM is headquartered in Armonk , NY .", "POSs":"NNP VBZ VBN IN NNP , NNP .","NERs":"ORG O O O LOC O LOC ."},
 {"text2":"Where is IBM located ?", "annotation2": {"toks":"Where is IBM located ?", "POSs":"WRB VBZ NNP VBN .","NERs":"O O ORG O O"},
 {"text3":"Armonk , NY", "annotation3": {"toks":"Armonk , NY", "POSs":"NNP , NNP","NERs":"LOC O LOC"}
}]
```
Text1 is the input sentence. Annotation1 is the POS annotation for the input sentence. Text2 is the desired question for the input. Annotation2 is the POS annotation for the question. Text3 is the target answer.
Since, We plan to do a version where target answer is hidden, there is some work to do to convert this dataset to make it appropriate for our purpose.

```
papers/ folder contains the reference papers using in this research.  
papers/question generation contains existing recent papers about the question generation task
papers/transformer-networks contains papers about transformer architecture and other improvements to it.
remaining papers in papers/ are about paraphrase identification
src/ folder contains all the source code. 
src/models contains LSTM and Convolutional models for paraphrase Identification
```

### References
[1] [Du, Xinya, Junru Shao, and Claire Cardie. "Learning to ask: Neural question generation for reading comprehension." arXiv preprint arXiv:1705.00106 (2017).](https://github.com/binays/sentence-pair-modeling/blob/master/papers/question%20generation/1705.00106.pdf)

[2] [Kim, Yanghoon, et al. "Improving neural question generation using answer separation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.](https://github.com/binays/sentence-pair-modeling/blob/master/papers/question%20generation/4629-Article%20Text-7668-1-10-20190707.pdf)

[3] [Duan, Nan, et al. "Question generation for question answering." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017.](https://github.com/binays/sentence-pair-modeling/blob/master/papers/question%20generation/D17-1090.pdf)

[4] [Sasazawa, Yuichi, Sho Takase, and Naoaki Okazaki. "Neural Question Generation using Interrogative Phrases." Proceedings of the 12th International Conference on Natural Language Generation. 2019.](https://github.com/binays/sentence-pair-modeling/blob/master/papers/question%20generation/W19-8613.pdf)
