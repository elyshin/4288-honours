import numpy as np
import json
import os

def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def test_train_split(data, ratio=0.5, seed=17):
    np.random.seed(seed)
    np.random.shuffle(data)
    split = int(ratio * len(data))
    return data[:split], data[split:]

if __name__ == '__main__':
    file_path = "../en_pud-ud-test.json"
    data = read_json(file_path)
    train_data, test_data = test_train_split(data)
    write_json("en_pud-ud-train.json", train_data)
    write_json("en_pud-ud-test.json", test_data)