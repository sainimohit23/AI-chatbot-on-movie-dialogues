# Chatbot on movie dialogues

In this project, I have built a chatbot using `seq2seq model or encoder-decoder model` model in TensorFlow. The objective of the model is to give replies to user inputs.
The model uses `attention mechanism` to learn.Model is written on latest tensorflow api.


I have trained this model for 240 epochs and it has learned some embeddings for most common words in the dataset. It still didn't learn to stop sequence after `<EOS>` token. And sometimes it did not output end of sentence token.
It may require around 1000 epochs to train this model completely. Due to limited resources I couldn't train it further.

Here are some of the responses model gave after 260 epochs of training:
![chatbot](https://user-images.githubusercontent.com/26195811/44617632-bbd0b700-a883-11e8-980e-79c9de65a5c0.png)



## *************************************************************************************




## Steps to use
### Data preprocessing
Extract the .rar file and run data_prepration.py. It will create a file named 'preprocess.p' which contains preprocessed text and dictionaries.

### Build model and Train
Use chatbotTrain.py to build the model and train it on dataset. Below are the steps involved in creating the model.

- __(1)__ define input parameters to the encoder model
  - `enc_dec_model_inputs`
- __(2)__ build encoder model
  - `encoding_layer`
- __(3)__ define input parameters to the decoder model
  - `enc_dec_model_inputs`, `process_decoder_input`, `decoding_layer`
- __(4)__ build decoder model for training
  - `decoding_layer_train`
- __(5)__ build decoder model for inference
  - `decoding_layer_infer`
- __(6)__ put (4) and (5) together
  - `decoding_layer`
- __(7)__ connect encoder and decoder models
  - `seq2seq_model`
- __(8)__ train and estimate loss and accuracy


![main-qimg-4af8f1e1933c5aa9f9f5a54838eedf98](https://user-images.githubusercontent.com/26195811/44617672-89738980-a884-11e8-97af-31cd3954c219.png)


Graph created on tensorboard:
![a](https://user-images.githubusercontent.com/26195811/44617682-b9229180-a884-11e8-8638-cdc9bf0b612c.png)


### Using chatbot
After training the model use bot.py to use it.


NOTE : To train model on google colab GPU, upload the file `.ipynb` file and `preprocess.p` on your google drive. Run .ipynb file to train.


Download variables trained for 260 epochs from [here](https://drive.google.com/open?id=1ne6GOcvZ0gxPD2jV0nOtWRif2b8Dyw9o)

## *************************************************************************************




