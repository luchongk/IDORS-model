# IDORS-model
Classification model used for IDORS project

The embeddings used for the baseline model are the ones provided at the fasttext website and can be downloaded from: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz. To use our model, download them and put them under the `models/fasttext_model` directory

If you want to use bert sentence embeddings, you'll also need to download a pre-trained model. You can find them in [this repository](https://github.com/google-research/bert) (section *Pre-trained models*). After you download it, put it under the `models/bert_model` directory.

The dependencies needed to run everything contained here are listed below (and can all be downloaded using pip):

  * fasttext
  * tensorflow (2.0 or higher)
  * [tweets-preprocessor](https://pypi.org/project/tweet-preprocessor/)
  * unidecode
  * sklearn
  * [bert-for-tf2](https://pypi.org/project/bert-for-tf2/)
