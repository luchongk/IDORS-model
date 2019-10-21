import csv
import preprocessor as p
import re
import unidecode
from random import shuffle
from math import floor

#TODO: -exclamation/question/ellipsis
#      -stopwords

""" class Dataset:
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def take(self, count):
        return Dataset(dataset.take(count), dataset.length - count)

    def skip(self, count):
        return Dataset(dataset.skip(count), dataset.length - count)
 """

def new_dataset(dataset_tsv_file, training_set_ratio):
    pairs = None

    with open('datasets/' + dataset_tsv_file) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        pairs = [(r['text'], r['HS']) for r in reader]

    shuffle(pairs)

    split_index = floor(training_set_ratio * len(pairs))

    training_set_file = open("training_set.txt", "w")
    test_set_file = open("test_set.txt", "w")
    
    for pair in pairs:
        result = "__label__"+ pair[1] + " " + preprocess(pair[0]) + "\n"
        if i < split_index:
            training_set_file.writelines(result)
        else:
            test_set_file.writelines(result)

def preprocess(tweet):
    p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.NUMBER)

    tweet = tweet.lower()
    tokenized = p.tokenize(tweet)
    leftFix = re.sub(r'(\S)(\$[^$\s]+?\$)', r'\1 \2', tokenized)
    rightFix = re.sub(r'(\$[^$\s]+?\$)(\S)', r'\1 \2', leftFix)

    return unidecode.unidecode(rightFix)

if __name__ == "__main__":
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

