import fasttext
import pprint
import sys

TRAINING_FILE = 'training_set.txt'
TEST_FILE = 'test_set.txt'

def train_and_test(retrain, save):
    if retrain:
        model = train(save)
    else:
        try:
            model = fasttext.load_model("models/fasttext_model/baseline.bin")
        except ValueError as err:
            print(err)
            print("Couldn't find a saved model, aborting...")
    
    if model:
        test(model)

def train(save):
    model = None
    try:
        model = fasttext.train_supervised(TRAINING_FILE, pretrainedVectors='models/fasttext_model/cc.es.300.vec', dim=300)
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
    
    train_and_test(retrain, save)