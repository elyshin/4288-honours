import json
from typing import List, Dict
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

def entity_filtering(sentences: List[Dict], threshold):
    for sentence in sentences:
        entity_sc = sentence['entity_sc']
        for entity in entity_sc:
            if entity['count'] < threshold:
                entity_sc.remove(entity)
    return sentences

def sample_filtering(sentences: List[Dict], threshold):
    for sentence in sentences:
        sample_sc = sentence['sample_sc']
        if sample_sc < threshold:
            sentences.remove(sentence)
    return sentences

def two_filtering(sentences: List[Dict], entity_threshold, sample_threshold):
    sentences = entity_filtering(sentences, entity_threshold)
    sentences = sample_filtering(sentences, sample_threshold)
    return sentences

def random_retrieval(sentences: List[Dict], k, seed=42):
    random.seed(seed)
    return random.sample(sentences, k)

def embeddings_array(sentences: List[Dict]):
    array = []
    for sentence in sentences:
        array.append(sentence['embedding'])
    array = np.array(array)
    return array

def knn_retrieval(input_sentence, sentences: List[Dict], k):
    input_embedding = input_sentence['embedding']
    embeddings = embeddings_array(sentences)
    knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
    distances, indices = knn.kneighbors([input_embedding])
    return [sentences[i] for i in indices[0]]