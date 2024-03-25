import argparse
import math
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
nltk.download('punkt')



def BP(gen, ref):
    l_gen = len(gen)
    l_ref = len(ref)
    if l_gen > l_ref:
        return 1
    else:
        return math.exp(1 - l_ref/l_gen)

def ngrams(tokens,n=1):
    l = len(tokens)
    grams = []
    for i in range(l - n + 1):
        grams.append(tuple(tokens[i:i + n]))
    return grams

def count_normal(tokens, ref,n=1):
    #print("count")
    gen_grams = ngrams(tokens, n=n)
    ref_grams = ngrams(ref, n=n)
    #print(gen_grams)
    #print(ref_grams)
    count = 0
    for gen_gram in gen_grams:
        for ref_gram in ref_grams : 
            if gen_gram == ref_gram:
                count += 1
                break
    #print("count : ",count)
    return count


def overlap_gram(tokens, n=1):
    l = len(tokens)

    grams = set()
    for i in range(l - n + 1):
        grams.add(tuple(tokens[i:i + n]))

    return grams

def count_clip(tokens, ref,n=1):
    #print("count_clip")
    max_ref = {}
    gen_grams = overlap_gram(tokens, n=n)
    ref_grams = ngrams(ref, n=n)
    #print(gen_grams)
    #print(ref_grams)
    for gen_gram in gen_grams:
        count = 0
        for ref_gram in ref_grams :
            
            if gen_gram == ref_gram :
                count +=1
            #print("{} {} | {}".format(gen_gram, ref_gram,max_ref))
        max_ref[gen_gram] = count
    #print("clip : ", max_ref)
    c = 0
    for v in max_ref:
        c += max_ref[v]
    
    return c

def cliped_precision(gen,ref,n):
    numer = count_clip(gen,ref,n)
    denom = len(ngrams(gen,n))
    if denom == 0:
        #print("{} / {} : {}".format(numer,denom,1))
        return 1
    else :
        #print("{} / {} : {}".format(numer,denom,numer/denom))
        return numer/denom
    
def recall(gen,ref,n):
    numer = count_clip(gen,ref,n)
    denom = len(ngrams(ref,n))

    #print("recall {} {}".format(numer,denom))
    return numer/denom
    
def LCS(gen,ref):
    n = 1
    l_gen = len(gen)
    l_ref = len(ref)

    max_c= 0
    max_seq = 0

    # Longest Common Sequence
    for i in range(l_gen):
        c = 0
        l_k = 0
        seq = []
        for j in range(i,l_gen) : 
            for k in range(l_k,l_ref) : 
                if gen[j] == ref[k] :
                    c+=1
                    l_k = k
                    seq.append(gen[j])
                    break
        if c > max_c:
            max_c = c
            max_seq = seq

    return max_c

# BLEU-4(bilingual evaluation understudy)
# https://en.wikipedia.org/wiki/BLEU
# https://wikidocs.net/31695
def BLEU(ref, gen,nGram):
    gen = word_tokenize(gen)
    ref = word_tokenize(ref)
    #print(gen)
    #print(ref)

    val = BP(gen,ref)
    precision = 1
    for n in range(1,nGram+1):
        precision*= cliped_precision(gen,ref,n)
    precision = precision**(1/nGram)
    val = val * precision

    return val

def ROUGE(ref,gen,nGram):
    gen = word_tokenize(gen)
    ref = word_tokenize(ref)
    #print(gen)
    #print(ref)

    precision = cliped_precision(gen,ref,nGram)
    v_recall = recall(gen,ref,nGram)
    if v_recall + precision == 0:
        return 0
    
    v_rouge = (2*precision*v_recall)/(precision+v_recall)

    return v_rouge

def ROUGE_L(ref,gen):
    gen = word_tokenize(gen)
    ref = word_tokenize(ref)
    #print(gen)
    #print(ref)

    lcs = LCS(gen,ref)
    p = lcs/len(ngrams(gen,1))
    r = lcs/len(ngrams(ref,1))
    return (2*p*r)/(p+r)

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ref", type=str)
    parser.add_argument("path_gen", type=str)
    args = parser.parse_args()

    ldx = 0

    t_BLUE4 = 0.0
    t_ROGUE2 = 0.0
    t_ROGUE_L =0.0


    with open(args.path_ref,"r") as f_ref:
        with open(args.path_gen,"r") as f_gen:
            l_ref = f_ref.readlines()
            l_gen = f_gen.readlines()
            for idx in range(len(l_gen)):
                ref = l_ref[idx]
                gen = l_gen[idx]
                #print("{} \n=>\n{}".format(gen,ref))
                t_BLUE4 += BLEU(ref,gen,4)
                t_ROGUE2 += ROUGE(ref,gen,2)
                t_ROGUE_L += ROUGE_L(ref,gen)

                ldx +=1

        
    print("Results – BLEU: {:.4f} ROUGE-2: {:.4f} ROUGE-L: {:.4f}".format(t_BLUE4/ldx,t_ROGUE2/ldx,t_ROGUE_L/ldx))
