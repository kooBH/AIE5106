import argparse
from nltk.tokenize import word_tokenize

# BLEU-4(bilingual evaluation understudy)
# https://en.wikipedia.org/wiki/BLEU
# https://wikidocs.net/31695
def BLUE(gen, ref,nGram):
    gen = word_tokenize(gen)
    ref = word_tokenize(ref)


    def clipped_precision(gen,ref,gram):
    
    
    p = 1
    for i in range(nGram):
        p *= clipped_precision(gen,ref,i+1)**(1/nGram)
    
    return min(1,len(gen)/len(ref))*p



if __name__ == "_main__" : 
    argparse.
    
    print()