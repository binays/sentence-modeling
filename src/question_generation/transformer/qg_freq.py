import numpy as np
from keras_transformer import get_model, decode
import json
from keras.preprocessing.text import text_to_word_sequence
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import tensorflow as tf
from keras.utils import multi_gpu_model
import nltk

source_tokens=[]
target_tokens=[]
test_x_tokens=[]
test_y_tokens=[]
count=1

#load training data
with open('/srv/binay/sentence_modeling/train_sent_pre.json') as f:
    data=json.load(f)

for d in data:
    target_answer=d['text3'].lower()
    text=d['text1'].lower()
    question=d['text2'].lower()

    text=text.replace(target_answer,'<MASKEDANSWER>')
    text=text_to_word_sequence(text)
    question=text_to_word_sequence(question)
    source_tokens.append(text)
    target_tokens.append(question)
    #if count==5000:
    #    break
    
    #count+=1

#load test data
with open('/srv/binay/sentence_modeling/test_sent_pre.json') as f:
    data=json.load(f)

for d in data:
    target_answer=d['text3'].lower()
    text=d['text1'].lower()
    question=d['text2'].lower()

    text=text.replace(target_answer,'<MASKEDANSWER>')
    text=text_to_word_sequence(text)
    question=text_to_word_sequence(question)
    test_x_tokens.append(text)
    test_y_tokens.append(question)


# Generate dictionaries
def build_token_dict(token_list):
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
    }
    for token in token_list:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
    return token_dict

frequency_dict={}
def build_freq_dict(token_list):
    for tokens in token_list:
        for token in tokens:
            if token not in frequency_dict:
                frequency_dict[token]=1
            else:
                frequency_dict[token]=frequency_dict[token]+1

build_freq_dict(source_tokens)
build_freq_dict(target_tokens)
build_freq_dict(test_x_tokens)
build_freq_dict(test_y_tokens)

print("Total tokens length:"+str(len(frequency_dict)))
freq_items=20000
most_freq_tokens=[]
count=1
for w in sorted(frequency_dict,key=frequency_dict.get,reverse=True):
    if count<=freq_items:
        most_freq_tokens.append(w)
    else:
        break
    count+=1

token_dict=build_token_dict(most_freq_tokens)
token_dict['<UNKNOWN>']=len(token_dict)

token_dict_inv = {v: k for k, v in token_dict.items()}

# Add special tokens
encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]
encode_test_x_tokens = [['<START>'] + tokens + ['<END>'] for tokens in test_x_tokens]

# Padding
source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))
test_x_max_len=max(map(len, encode_test_x_tokens))

if source_max_len>test_x_max_len:
    encode_max_len=source_max_len
else:
    encode_max_len=test_x_max_len

encode_tokens = [tokens + ['<PAD>'] * (encode_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

encode_test_x_tokens = [tokens + ['<PAD>'] * (encode_max_len - len(tokens)) for tokens in encode_test_x_tokens]



def get_input(tokens,flag):
    input_lst=[]
    for et in tokens:
        each=[]
        for t in et:
            if t not in token_dict:
                if flag==1:
                    each.append([token_dict['<UNKNOWN>']])
                else:
                    each.append(token_dict['<UNKNOWN>'])
            else:
                if flag==1:
                    each.append([token_dict[t]])
                else:
                    each.append(token_dict[t])
        input_lst.append(each)
    return input_lst 

encode_input=get_input(encode_tokens,0)
decode_input=get_input(decode_tokens,0)
decode_output=get_input(output_tokens,1)
encode_test_input=get_input(encode_test_x_tokens,0)

#print(encode_input[0])
#print(decode_input[0])
##print(decode_output[0])
print(encode_test_input[0])

#print("encode input")
#print(encode_input)
#print("decode input")
#print(decode_input)
#print("decode output")
#print(decode_output)

# Build & fit model
with tf.device("/cpu:0"):
    model_cpu = get_model(
        token_num=len(token_dict),
        embed_dim=32,
        encoder_num=6,
        decoder_num=6,
        head_num=8,
        hidden_dim=128,
        dropout_rate=0.05,
        use_same_embed=True,  # Use different embeddings for different languages
    )
model=multi_gpu_model(model_cpu,gpus=4)
model.compile('adam', 'sparse_categorical_crossentropy')
model.summary()

model.fit(
    x=[np.array(encode_input), np.array(decode_input)],
    y=np.array(decode_output),
    epochs=10,
    batch_size=128,
)

# Predict
decoded = decode(
    model,
    encode_test_input,
    start_token=token_dict['<START>'],
    end_token=token_dict['<END>'],
    pad_token=token_dict['<PAD>'],
)

predicted_y=[]

for d_x in decoded:
    os=' '.join(map(lambda x: token_dict_inv[x], d_x[1:-1]))
    os=os.split(' ')
    predicted_y.append([os])

bleu_score=nltk.translate.bleu_score.corpus_bleu(predicted_y,test_y_tokens,weights=(0.5,0.5))
print(bleu_score)


