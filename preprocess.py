import csv
import preprocessor as p
import re
import unidecode
from random import shuffle
from math import floor

#TODO: -exclamation/question/ellipsis
#      -stopwords

TRAINING_RATIO = 0.8
VALIDATION_RATIO = 1 - TRAINING_RATIO

p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.NUMBER)

with open('datasets/dev_es.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    pairs = [(r['text'], r['HS']) for r in reader]

    shuffle(pairs)
    
    split_index = floor(TRAINING_RATIO * len(pairs))

    training_set = open("training_set.txt", "w")
    test_set = open("test_set.txt", "w")
    
    for i, (tweet, label) in enumerate(pairs):
        tweet = tweet.lower()
        tokenized = p.tokenize(tweet)
        leftFix = re.sub(r'(\S)(\$[^$\s]+?\$)', r'\1 \2', tokenized)
        rightFix = re.sub(r'(\$[^$\s]+?\$)(\S)', r'\1 \2', leftFix)
        
        result = "__label__"+ label + " " + unidecode.unidecode(rightFix)+ "\n"
        
        if i < split_index:
            training_set.writelines(result)
        else:
            test_set.writelines(result)

    training_set.close()
    test_set.close()