import fasttext
import pprint
import sys

from data_mgmt.data_mgmt import Dataset

#TODO: 

def train_and_test(dataset, training_set_ratio, retrain, save):
    train_set = dataset.take(training_set_ratio)

def get_ft_model(training_set=None, test_set=None, save=False):
    model = None
    if not training_set:
        try:
            model = fasttext.load_model("baseline.bin")
        except ValueError as err:
            print(err)
            print("Couldn't find a saved model, aborting...")    
    else:
        model = train_model(training_set, test_set, save)

    return model

def train_model(training_set, test_set, save):
    model = None
    training_set_fn = 'ft_training_set.txt'
    test_set_fn = 'ft_test_set.txt'

    try:
        write_dataset_to_file(training_set_fn, training_set)
        write_dataset_to_file(test_set_fn, test_set)
        model = fasttext.train_supervised(training_set_fn, pretrainedVectors='models/cc.es.300.vec', dim=300)
        if save:
            model.save_model("baseline.bin")
    except Exception as err:
        print(err)

    return model

def test_model(model, test_set_fn):
    pprint.pprint(model.test(test_set_fn))
    pprint.pprint(model.test_label(test_set_fn))

def write_dataset_to_file(file_name, dataset):
    with open(file_name, 'w') as dataset_file:
        for (tweet, label) in dataset:
            dataset_file.writelines("__label__"+ label + " " + tweet + "\n")

if __name__ == "__main__":
    execute(sys.argv[1], sys.argv[2])