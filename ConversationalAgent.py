import json
import random
from abc import ABCMeta, abstractmethod

import numpy as np
import spacy
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nlp = spacy.load("es_core_news_sm")


class ConversationalAgent(metaclass=ABCMeta):

    @abstractmethod
    def train_model(self):
        """ Implemented in child class """

    @abstractmethod
    def request(self, message):
        """ Implemented in child class """

    @abstractmethod
    def save_model(self, model_name=None):
        """ Implemented in child class """

    @abstractmethod
    def load_model(self, model_name=None):
        """ Implemented in child class """

    @abstractmethod
    def _predict_class(self, sentence):
        """ Implemented in child class """


class CboConversationalAgent(ConversationalAgent):

    def __init__(self, intents, model_name="chatbot"):
        self.classes = None
        self.words = None
        self.model = None
        self.hist = None
        self.intents = intents
        self.tokenizer = None
        self.model_name = model_name

        if intents.endswith(".json"):
            print(intents)
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())

    def train_model(self):
        train_y = []
        train_x = []

        sentences = []
        labels = []
        documents = []

        self.classes = []

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                sentences.append(pattern)
                labels.append(intent['tag'])
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(sentences)
        word_index = self.tokenizer.word_index
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, maxlen=5)

        print(padded)
        print(word_index)

        for i in range(len(padded)):
            documents.append([padded[i], labels[i]])

        random.shuffle(documents)
        output_empty = [0] * len(self.classes)

        for document in documents:
            train_x.append(document[0])
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            train_y.append(output_row)

        print(train_x)
        print(train_y)

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(100, 16, input_length=len(train_x[0])),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.hist = self.model.fit(train_x, train_y, epochs=120,
                                   batch_size=16,
                                   verbose=1)

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"{self.model_name}.h5", self.hist)
            json.dump(self.classes, open(f'{self.model_name}_classes.json', 'w'))
        else:
            self.model.save(f"{model_name}.h5", self.hist)
            json.dump(self.classes, open(f'{model_name}_classes.json', 'w'))

    def load_model(self, model_name=None):
        if model_name is None:
            self.classes = json.load(open(f'{self.model_name}_classes.json', 'r'))
            self.model = load_model(f'{self.model_name}.h5')
        else:
            self.classes = json.load(open(f'{model_name}_classes.json', 'r'))
            self.model = load_model(f'{model_name}.h5')

    def _predict_class(self, sentence):
        sent = []
        sent.append(sentence)
        p = self.tokenizer.texts_to_sequences(sent)
        padded = pad_sequences(p, maxlen=5)
        print(padded)
        res = self.model.predict(np.array(padded))[0]
        print(res)
        print(np.argmax(res))
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        print(return_list)
        return return_list

    def _get_response(self, ints, intents_json):
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "Lo siento, áun estoy aprendiendo, ¿puedes repetir la pregunta?"
        return result

    def request(self, message):
        ints = self._predict_class(message)
        print(self._get_response(ints, self.intents))
        return self._get_response(ints, self.intents)
