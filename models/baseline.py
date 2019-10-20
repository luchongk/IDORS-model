import fasttext
import pprint

retrain = True
save = True
model = None
try:
    if retrain:
        model = fasttext.train_supervised('training_set.txt', pretrainedVectors='cc.es.300.vec', dim=300)
        if save:
            model.save_model("baseline.bin")
    else:
        model = fasttext.load_model("baseline.bin")
except ValueError as err:
    print("Couldn't find a saved model, aborting...")

pprint.pprint(model.test("test_set.txt"))

pprint.pprint(model.test_label("test_set.txt"))