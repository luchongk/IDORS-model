import bert, configparser, os

from tensorflow import keras

def get_bert_layer(bert_model_dir):
    bert_params = bert.params_from_pretrained_ckpt(bert_model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, trainable=False, name="bert")
    
    return l_bert

def get_bert_model(bertTokensShape):
    config = configparser.ConfigParser()
    config.read('conf.txt')
    bert_model_dir = config['GENERAL']['BERT_MODEL_DIR']
    bert_ckpt = config['GENERAL']['BERT_CKPT']

    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    bert_model_dir = os.path.join(current_dir, "bert_model" ,bert_model_dir)

    inputs = keras.Input(shape=bertTokensShape, name='bert_token_ids')

    bert_layer = get_bert_layer(bert_model_dir)
    bert_vectors = bert_layer(inputs)
    bert_vectors = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_vectors)

    model = keras.Model(inputs=inputs, outputs=bert_vectors, name="bert_vectors")

    bert_ckpt_file = os.path.join(bert_model_dir, bert_ckpt)
    bert.load_stock_weights(bert_layer, bert_ckpt_file)

    return model