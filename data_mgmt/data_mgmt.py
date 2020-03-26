import csv, re, unidecode, nltk 
import preprocessor as p, numpy as np

from random import shuffle
from math import floor
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

#TODO: tokenize LOL
#TODO: John lennon

MAX_WORDS = 33

def new_dataset(dataset_tsv_file, training_set_ratio):
    pairs = None

    with open('datasets/' + dataset_tsv_file) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        pairs = [(r['text'], r['HS']) for r in reader]

    shuffle(pairs)

    split_index = floor(training_set_ratio * len(pairs))

    training_set_file = open("training_set.txt", "w")
    test_set_file = open("test_set.txt", "w")
    
    training_words = set()
    test_words = set()
    for i, pair in enumerate(pairs):
        preprocessed = preprocess(pair[0])
        result = "__label__"+ pair[1] + " " + preprocessed + "\n"
        if i < split_index:
            training_words = training_words.union(preprocessed.split())
            training_set_file.writelines(result)
        else:
            test_words = test_words.union(preprocessed.split())
            test_set_file.writelines(result)

    training_set_file.close()
    test_set_file.close()

    wordsNewInTest = test_words - training_words
    wordRatio = len(wordsNewInTest) / len(test_words)    
    pass


def preprocess(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'\\n', ' ', tweet)
    tweet = re.sub(r'(\S)(https?):', r'\1 \2:', tweet)
    p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG)
    tweet = p.tokenize(tweet)
    
    tokenizer = nltk.tokenize.TweetTokenizer()
    tweet = tokenizer.tokenize(tweet)
    tweet = ' '.join(tweet)
    tweet = re.sub(r'\$ ([A-Z]+?) \$', r'$\1$', tweet)
    
    tweet = tweet.split(' '); 
    ### Stopwords removal ###
    stop_words = set(stopwords.words('spanish'))
    new_sentence = []
    for w in tweet:
        if w not in stop_words:
            new_sentence.append(w)
    tweet = ' '.join(new_sentence)

    tweet = unidecode.unidecode(tweet)

    p.set_options(p.OPT.NUMBER)
    tweet = p.tokenize(tweet)
    tweet = re.sub(r'([!¡]\s?){3,}', r' $EXCLAMATION$ ', tweet)
    tweet = re.sub(r'([¿?]\s?){3,}', r' $QUESTION$ ', tweet)
    tweet = re.sub(r'(\.\s?){3,}', r' $ELLIPSIS$ ', tweet)

    return tweet
    #tweet = re.sub(r'(\S)(\$[^$\s]+?\$)', r'\1 \2', tweet)
    #tweet = re.sub(r'(\$[^$\s]+?\$)(\S)', r'\1 \2', tweet)
    #return unidecode.unidecode(rightFix)

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
    result = np.empty((len(dataset), MAX_WORDS, ft_model.get_dimension()))

    for index, example in enumerate(dataset):
        result[index] = get_tweet_embeddings(example[0], ft_model)

    return result

def get_word_embedding(word, ft_model):
    return ft_model.get_word_vector(word)

def get_tweet_embeddings(tweet, ft_model):
    vecs = np.zeros((MAX_WORDS, ft_model.get_dimension()))
    words = tweet.split()
    for i in range(MAX_WORDS):
        if i < len(words):
            vecs[i] = get_word_embedding(words[i], ft_model)
    return vecs