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
    #print(count)
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
        max_ref[gen_gram] = count
    #print(max_ref)
    c = 0
    for v in max_ref:
        c += max_ref[v]
    
    return c


# BLEU-4(bilingual evaluation understudy)
# https://en.wikipedia.org/wiki/BLEU
# https://wikidocs.net/31695
def BLEU(ref, gen,nGram):
    gen = word_tokenize(gen)
    ref = word_tokenize(ref)

    def cliped_precision(gram,ref,n):
        numer = count_clip(gram,ref,n)
        denom = count_normal(gram,ref,n)
        if denom == 0:
            return 1
        else :
            return numer/denom

        #print("{} / {} : {}".format(numer,denom,numer/denom))
    val = BP(gen,ref)
    precision = 1
    for n in range(1,nGram+1):
        precision*= cliped_precision(gen,ref,n)
    precision = precision**(1/nGram)
    val = val * precision

    return val



if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument("file_in", type=str)
    parser.add_argument("file_out", type=str)
    args = parser.parse_args()

    ldx = 0

    with open(args.file_in,"r") as f_ref:
        with open(args.file_out,"r") as f_gen:
            l_ref = f_ref.readlines()
            l_gen = f_gen.readlines()
            for idx in range(len(l_gen)):
                ref = l_ref[idx]
                gen = l_gen[idx]
                print("{} => {}".format(gen,ref))
                print(BLEU(ref,gen,4))

                import nltk.translate.bleu_score as bleu

                print(bleu.sentence_bleu([ref],gen))

                if ldx > 0 : 
                    break
                ldx +=1

