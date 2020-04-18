import fasttext, pprint, sys, configparser

TRAINING_FILE = 'training_set.txt'
TEST_FILE = 'test_set.txt'

def train_and_test(retrain, save, language):
    if retrain:
        model = train(save, language)
    else:
        try:
            model = fasttext.load_model("models/fasttext_model/baseline.bin")
        except ValueError as err:
            print(err)
            print("Couldn't find a saved model, aborting...")
    
    if model:
        test(model)

def train(save, language):
    model = None

    if language == 'english':
        vec_file = 'models/fasttext_model/cc.en.300.vec'
    elif language == 'spanish':
        vec_file = 'models/fasttext_model/cc.es.300.vec'
    else:
        print("Language {language} not supported.")
        exit(0)
    
    try:
        model = fasttext.train_supervised(TRAINING_FILE, pretrainedVectors=vec_file, dim=300)
        if save:
            model.save_model("models/fasttext_model/baseline.bin")
    except Exception as err:
        print(err)

    return model

def test(model):
    pprint.pprint(model.test(TEST_FILE))
    pprint.pprint(model.test_label(TEST_FILE))

if __name__ == "__main__":
    retrain = True if '--retrain' in sys.argv else False 
    save = True if '--save' in sys.argv else False
    
    config = configparser.ConfigParser()
    config.read('conf.txt')

    language = config['GENERAL']['LANGUAGE']

    train_and_test(retrain, save, language)