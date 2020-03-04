import csv, re, unidecode, nltk 
import preprocessor as p, numpy as np

from random import shuffle
from math import floor
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def new_dataset(dataset_tsv_file, training_set_ratio):
    pairs = None

    with open('datasets/' + dataset_tsv_file) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        pairs = [(r['text'], r['HS']) for r in reader]

    shuffle(pairs)

    split_index = floor(training_set_ratio * len(pairs))

    training_set_file = open("training_set.txt", "w")
    test_set_file = open("test_set.txt", "w")
    
    for i, pair in enumerate(pairs):
        result = "__label__"+ pair[1] + " " + preprocess(pair[0]) + "\n"
        if i < split_index:
            training_set_file.writelines(result)
        else:
            test_set_file.writelines(result)

    training_set_file.close()
    test_set_file.close()

def preprocess(tweet):
    tweet = tweet.lower()
    
    ### Stopwords removal ###
    tweet = nltk.word_tokenize(tweet)
    stop_words = set(stopwords.words('spanish'))
    new_sentence = []
    for w in tweet:
        if w not in stop_words:
            new_sentence.append(w)
    tweet = ' '.join(new_sentence)

    p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.NUMBER)
    tokenized = p.tokenize(tweet)
    tokenized = re.sub(r'([!¡]\s?){3,}', r' $EXCLAMATION$ ', tokenized)
    tokenized = re.sub(r'([¿?]\s?){3,}', r' $QUESTION$ ', tokenized)
    tokenized = re.sub(r'(\.\s?){3,}', r' $ELLIPSIS$ ', tokenized)
    leftFix = re.sub(r'(\S)(\$[^$\s]+?\$)', r'\1 \2', tokenized)
    rightFix = re.sub(r'(\$[^$\s]+?\$)(\S)', r'\1 \2', leftFix)

    return unidecode.unidecode(rightFix)

def get_dataset():
    parsing_regex = re.compile(r'^__label__(\d)\s{1}(.*)$')

    train_file = open('training_set.txt')
    test_file = open('test_set.txt')

    training_dataset = []
    test_dataset = []

    for line in train_file:
        match = re.match(parsing_regex, line)
        training_dataset.append((match[2], match[1]))

    for line in test_file:
        match = re.match(parsing_regex, line)
        test_dataset.append((match[2], match[1]))

    return training_dataset, test_dataset

def dataset_to_embeddings(dataset, ft_model):
    result = np.empty((len(dataset), ft_model.get_dimension()))

    for index, example in enumerate(dataset):
        result[index] = get_tweet_embedding(example[0], ft_model)

    return result
    
def get_tweet_embedding(tweet, ft_model):
    vec = np.repeat(0.0, ft_model.get_dimension())
    words = tweet.split()
    for word in words:
        vec += np.array(ft_model.get_word_vector(word))
    return np.divide(vec, len(words))