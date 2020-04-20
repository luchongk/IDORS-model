import csv, re, unidecode, nltk, os, bert, configparser, sys
import preprocessor as p, numpy as np

from random import shuffle
from math import floor
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('punkt')
nltk.download('stopwords')

#TODO: Separate dataset and vector creation

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

    bert_training_file = open("bert_training_set.txt", "w")
    bert_test_file = open("bert_test_set.txt", "w")
    
    training_words = set()
    test_words = set()
    for i, pair in enumerate(pairs):
        preprocessed = preprocess(pair[0])
        result = "__label__"+ pair[1] + " " + preprocessed + "\n"
        bp_tweet = bert_preprocess(pair[0]) + "\n"
        if i < split_index:
            training_words = training_words.union(preprocessed.split())
            training_set_file.writelines(result)
            bert_training_file.writelines(bp_tweet)
        else:
            test_words = test_words.union(preprocessed.split())
            test_set_file.writelines(result)
            bert_test_file.writelines(bp_tweet)

    training_set_file.close()
    test_set_file.close()
    
    bert_training_file.close()
    bert_test_file.close()

    wordsNewInTest = test_words - training_words
    wordRatio = len(wordsNewInTest) / len(test_words)

def create_bert_tokenizer():
    config = configparser.ConfigParser()
    config.read('conf.txt')
    bert_model_dir = config['GENERAL']['BERT_MODEL_DIR']

    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    models_folder = os.path.join(current_dir, "models/bert_model", bert_model_dir)
    vocab_file = os.path.join(models_folder, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

def bert_preprocess(tweet):
    b_tweet = tweet.lower()
    b_tweet = re.sub(r'\\n', ' ', b_tweet)
    b_tweet = re.sub(r'(\S)(https?):', r'\1 \2:', b_tweet)
    p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG)
    b_tweet = p.clean(b_tweet)

    b_tweet = unidecode.unidecode(b_tweet)
    b_tweet = re.sub(r'([\w\d]+)([^\w\d ]+)', r'\1 \2', b_tweet)
    b_tweet = re.sub(r'([^\w\d ]+)([\w\d]+)', r'\1 \2', b_tweet)

    return b_tweet

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
    language = os.getenv('LANGUAGE')
    stop_words = set(stopwords.words(language))
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
    all_tweets = []

    for line in train_file:
        match = re.match(parsing_regex, line)
        all_tweets.append(match[2])
        training_dataset.append((match[2], match[1]))

    for line in test_file:
        match = re.match(parsing_regex, line)
        all_tweets.append(match[2])
        test_dataset.append((match[2], match[1]))
    
    extra_embeddings = get_additional_embeddings(all_tweets)

    training_ex_emb = extra_embeddings[0:len(training_dataset)]
    test_ex_emb = extra_embeddings[-len(test_dataset):]

    return training_dataset, test_dataset, training_ex_emb, test_ex_emb

def get_bert_token_ids():
    bert_tokenizer = create_bert_tokenizer()
    train_ids = []
    test_ids = []

    if not (os.path.exists('bert_training_ids.txt') and os.path.exists('bert_test_ids.txt')):
        with open("bert_training_set.txt") as tweets_file:
            tweets = tweets_file.readlines()
            for tweet in tweets:
                train_tokens = ["[CLS]"] + bert_tokenizer.tokenize(tweet) + ["[SEP]"]
                train_token_ids = bert_tokenizer.convert_tokens_to_ids(train_tokens)
                train_ids.append(train_token_ids)

        with open("bert_test_set.txt") as tweets_file:
            tweets = tweets_file.readlines()
            for tweet in tweets:
                test_tokens = ["[CLS]"] + bert_tokenizer.tokenize(tweet) + ["[SEP]"]
                test_token_ids = bert_tokenizer.convert_tokens_to_ids(test_tokens)
                test_ids.append(test_token_ids)
        max_train_length = max(map(len, train_ids))
        max_test_length = max(map(len, test_ids))

        max_length = max(max_train_length, max_test_length)

        train_ids = np.array([ id_list + [0] * (max_length - len(id_list)) for id_list in train_ids])
        test_ids = np.array([ id_list + [0] * (max_length - len(id_list)) for id_list in test_ids])

        np.savetxt('bert_training_ids.txt', train_ids)
        np.savetxt('bert_test_ids.txt', test_ids)
    else:
        train_ids = np.loadtxt('bert_training_ids.txt')
        test_ids = np.loadtxt('bert_test_ids.txt')

    return train_ids, test_ids

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

def get_additional_embeddings(all_tweets):
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_embeddings = tf_idf_vectorizer.fit_transform(all_tweets)

    svd = TruncatedSVD(n_components=100)
    reduced_dimension_embeddings = svd.fit_transform(tf_idf_embeddings)

    return reduced_dimension_embeddings

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('conf.txt')

    training_set_ratio = float(config['GENERAL']['TRAINING_SET_RATIO'])
    dataset_name = config['GENERAL']['DATASET_NAME']

    os.environ['LANGUAGE'] = config['GENERAL']['LANGUAGE']

    new_dataset(dataset_name, training_set_ratio)
    