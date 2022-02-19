# coding: utf-8
import pickle

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def save_events(mabed_object, file_path):
    with open(file_path, 'wb') as output_file:
        pickle.dump(mabed_object, output_file)


def load_events(file_path):
    with open(file_path, 'rb') as input_file:
        return pickle.load(input_file)


def load_stopwords(file_path, code):
    stopwords = set()
    with open(file_path, 'r', encoding=code) as input_file:
        for line in input_file.readlines():
            stopwords.add(line.strip('\n'))
    return stopwords
