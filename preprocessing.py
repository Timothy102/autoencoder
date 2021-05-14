import tensorflow as tf
import unicodedata
import re
import numpy as np

class DataLoader():
    def __init__(self, path):
        self.path = path
        
    def load_data():
        # Run this cell to load the dataset
        NUM_EXAMPLES = 20000
        data_examples = []
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                if len(data_examples) < NUM_EXAMPLES:
                    data_examples.append(line)
                else:
                    break

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"ü", 'ue', sentence)
        sentence = re.sub(r"ä", 'ae', sentence)
        sentence = re.sub(r"ö", 'oe', sentence)
        sentence = re.sub(r'ß', 'ss', sentence)
        
        sentence = unicode_to_ascii(sentence)
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r"[^a-z?.!,']+", " ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        
        return sentence.strip()


