import math

from nltk import word_tokenize
from collections import Counter
from nltk.util import ngrams
import sys
import string
def load_dict(file_name):
    word_dict={}
    f=open(file_name,'r')
    for line in f:
        lines=line.strip().split('\t')
        if(len(lines)==2):
            word_dict[lines[0]]=int(lines[1])
    return word_dict
def sen_to_array(sen,word_dict,sen1):
    sens=sen.strip().split(' ')
    sen1s=sen1.strip().split(' ')
    words=[]
    for i in sens:
        '''
        if(i not in sen1s):
            words.append(len(word_dict)+1)
            continue
        '''
        if(word_dict.get(i)!=None):
            words.append(word_dict.get(i))
    return words
def compute(candidate, references, weights):
    #candidate = [c.lower() for c in candidate]
    #references = [[r.lower() for r in reference] for reference in references]
    p_ns = (modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))
        
    s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)

    bp = brevity_penalty(candidate, references)
    return bp * math.exp(s)

def modified_precision(candidate, references, n):
    counts = counter_gram(candidate, n)
    #print counts

    if not counts:
        return 0
    max_counts = {}
    for reference in references:
        reference_counts = counter_gram(reference, n)
        for ngram in counts.keys():
            if(reference_counts.get(ngram)!=None):
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])
            else:
                max_counts[ngram] = 0.000000001
    clipped_counts = dict((ngram, min(counts[ngram], max_counts[ngram])) for ngram in counts.keys())
    #print counts
    #print clipped_counts

    return sum(clipped_counts.values()) / sum(counts.values())
def counter_gram(word_array,n):
    ngram_words={}
    for i in range(0,len(word_array)-n+1):
        tmp_i=''
        for j in range(0,n):
            tmp_i+=str(word_array[i+j])
            tmp_i+=' '
        if(ngram_words.get(tmp_i)==None):
            ngram_words[tmp_i]=1
        else:
            ngram_words[tmp_i]+=1
    return ngram_words    


def brevity_penalty(candidate, references):
    c = len(candidate)
    r = min(abs(len(r) - c) for r in references)
    if c==0:
        return 0
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)
#evaluation/outputs/yelp/sentiment.test.0.cbert   evaluation/outputs/yelp/sentiment.test.0.human  test_tools/li_test_tool/BLEU/zhi.dict.orgin 4

def eval_bleu(generate_file, orgin_file, dict_file="test_tools/li_test_tool/BLEU/zhi.dict.orgin", weight_num=4):
    word_dict=load_dict(dict_file) #dict_file
    can={}
    query=''
    answer=[]
    #weight=[0.5,0.5]
    weight=[]
    for i in range(weight_num):
        weight.append(1.0/weight_num)
    f=open(generate_file,'r') #generate_file
    for line in f:
        lines=line.strip().split('\t')
        if(len(lines)==3):
            can[lines[0].strip()]=sen_to_array(lines[1].strip(),word_dict,lines[0])
            '''
            if(lines[1]=='1'):
                if(query!='' and answer!=[]):
                    can[query]=sen_to_array(answer[0].strip(),word_dict)
                query=lines[0]
                answer=[]
            else:
                answer.append(lines[0].replace('result: <END> ',''))
            '''
    f.close()
    #print(len(can))
    ref={}
    f=open(orgin_file,'r') #orgin_file
    for line in f:
        lines=line.strip().split('\t')
        if(len(lines)==3):
            lines[0]=lines[0]
            if(ref.get(lines[0].strip())==None):
                tmp=[]
                tmp.append(sen_to_array(lines[1].strip(),word_dict,lines[0]))
                ref[lines[0].strip()]=tmp
            else:
                ref[lines[0].strip()].append(sen_to_array(lines[1].strip(),word_dict,lines[0]))
    f.close()
    #print(len(ref))
    #print len(can.keys())
    #print ref.keys()
    #print len(ref.keys())
    bleu_array=[]
    bleu_total=0
    for i in can.keys():
        if(ref.get(i)!=None):
            #print 'ok'
            bleu_score=compute(can[i],ref[i],weight)
            bleu_total+=bleu_score
            bleu_array.append(bleu_score)
            #print can[i]
            #print ref[i]
    #print(len(bleu_array))
    return bleu_total/len(bleu_array)

if __name__ == "__main__":

    task_name = sys.argv[1]
    model_name = sys.argv[2]
    dict_file = "test_tools/li_test_tool/BLEU/zhi.dict.orgin"
    weight_num = 4

    generate_file = "evaluation/outputs/{}/sentiment.test.0.{}".format(task_name, model_name)
    orgin_file = "evaluation/outputs/{}/sentiment.test.0.human".format(task_name)
    blue_0 = eval_bleu(generate_file, orgin_file, dict_file, weight_num)

    generate_file = "evaluation/outputs/{}/sentiment.test.1.{}".format(task_name, model_name)
    orgin_file = "evaluation/outputs/{}/sentiment.test.1.human".format(task_name)
    blue_1 = eval_bleu(generate_file, orgin_file, dict_file, weight_num)

    print("bleu:{}".format((blue_0 + blue_1) / 2))