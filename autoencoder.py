import tensorflow as tf
import unicodedata
import re
import numpy as np
import json

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config import *

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=str, default=5,
                        help="Epochs to train the model.")
    args = parser.parse_args()
    return args



for es in english_sentences:
    es=preprocess_sentence(es)
    
for gs in german_sentences:
    gs=start_token+preprocess_sentence(gs)+end_token

tokenizer = Tokenizer(filters='')


tokenizer.fit_on_texts(german_sentences)
config = tokenizer.get_config()
word_index = json.loads(config['word_index'])
index_words  = json.loads(config['index_word'])

num_samples =5
inx = np.random.choice(len(english_sentences),num_samples, replace=False)
print(inx)

sequences=tokenizer.texts_to_sequences(german_sentences)
padded = pad_sequences(sequences, padding='post',value=0)

def map_embedding_f(x,y):
    inp=[]
    pad=tf.pad(x, paddings=[[13-tf.shape(x)[0],0][0,0]], mode='CONSTANT')
    inp.append(pad)
    inp.append(y)
    return inp[0], inp[1]


def load_data():
    from sklearn.model_selection import train_test_split

    x_english_train, x_english_test = train_test_split(english_sentences, test_size=0.2)
    x_german_train, x_german_test = train_test_split(sequences, test_size=0.2)


    x_english_train = np.array(x_english_train)
    x_english_test = np.array(x_english_test)
    x_german_train = np.array(x_german_train)
    x_german_test = np.array(x_german_test)

    trainDataset = tf.data.Dataset.from_tensor_slices((x_english_train,x_german_train))
    validationDataset = tf.data.Dataset.from_tensor_slices((x_english_test,x_german_test ))


    print(trainDataset.element_spec)
    print(validationDataset.element_spec)

def load_model():
    emb_layer = tf.keras.models.load_model('./models/tf2-preview_nnlm-en-dim128_1')
    emb_layer(tf.constant(["these", "aren't", "the", "droids", "you're", "looking", "for"])).shape
    sequences = tokenizer.texts_to_sequences(german_sentences)



from tensorflow.keras.layers import Layer
class CustomLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.learned_emb = tf.Variable(initial_value=tf.zeros(shape=(1,128),dtype=tf.float32))
        
    def call(self, inputs):
        x=self.learned_emb
        x=tf.tile(x,[tf.shape(inputs)[0],1])
        x=tf.expand_dims(x,axis=1, name=None)
        return tf.keras.layers.Concatenate(axis=1)([inputs,x])
    
myLayer = CustomLayer()

from tensorflow.keras.layers import Dense, Flatten, Masking, LSTM
from tensorflow.keras import Input, Model

inputs = Input(batch_shape = (None, 13, 128))
x = myLayer(inputs)
x = Masking(mask_value = 4.0)(x)
hidden_state, cell_state = LSTM(512, return_state=True, name='stateful')(x)
encoder = Model(inputs = inputs, outputs = [hidden_state, cell_state])


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout


class MyDecoder(Model):
    def __init__(self):
        super(MyDecoder,self).__init__()
        self.embedding=Embedding(len(tokenizer.word_index),128,mask_zero=True)
        self.lstm=LSTM(512,return_sequences=True, return_state=True)
        self.dense=Dense(len(tokenizer.word_index))
    def call(self,inputs, hidden_state=None,cell_state=None):
        x=self.embedding(inputs)
        if hidden_state is not None and cell_state is not None:
            x, hidden_state, cell_state=self.lstm(x, initial_state=[hidden_state, cell_state])
        else:
            x, hidden_state, cell_state=self.lstm(x)
            x=self.dense(x)
        return x, hidden_state, cell_state
        
decoder=MyDecoder()

hidden, cell = Encoder(next(iter(trainDataset.take(1)))[0])
print(decoder(next(iter(trainDataset.take(1))))[1],hidden, cell)

def convert(data):
    german_inputs = tf.zeros([data.shape[0],data.shape[1], tf.int32])
    german_outputs = tf.zeros([data.shape[0],data.shape[1], tf.int32])
    german_inputs=data
    german_inputs = tf.where(german_inputs[:]==word_index.get(end_token),x=0,y=german_inputs)
    german_outputs = tf.where(german_inputs[:]==word_index.get(start_token),x=0,y=german_inputs)

    return german_inputs, german_outputs

optimizer = tf.keras.optimizers.RMSprop(0.05)
loss = tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True)
variables = Encoder.trainable_variables+ Decoder.trainable_variables

@tf.function
def grad_fn(data1, data, german_outputs, loss):
    with tf.GradientTape() as tape:
        x,y = Encoder(data1)
        a,b,c = Decoder(data, x, y)
        loss_value=loss(german_outputs, loss)
    return loss_value, tape.gradient(loss_value, variables)
        

def train_autoencoder(num_epochs, dataset, val_dataset, val_steps, grad_fn, optimizer, loss):
    train_loss_results=[]
    val_loss_results = []
    
    for epoch in range(num_epochs):
        epoch_loss_avg=tf.keras.metrics.Mean()
        val_epoch_loss_avg = tf.keras.metrics.Mean()
        
        for x,y in dataset:
            inp, out = convert(y)
            loss_value, grads = grad_fn(x,y,out, loss)
            optimizer.apply_gradients(zip(grads, variables))
            
            epoch_loss_avg(loss_value)
        for a,b in val_dataset:
            preds,_,_ = decoder(b)
            val_epoch_loss_avg(loss(out, preds))
        train_loss_results.append(epoch_loss_avg.result())
        val_loss_results.append(val_epoch_loss_avg.result())
        
        print(" Epoch :{03d}: Loss: {:.3f} : Val_loss: {:.3f}".format(epoch, epoch_loss_results(), val_epoch_loss_results()))
        
    return train_loss_results, val_loss_results

def main():
    args = parseArguments()
    model = load_model()
    history = train_autoencoder(model, args.epochs, args.batch_size)