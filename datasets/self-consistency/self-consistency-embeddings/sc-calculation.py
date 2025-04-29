import json
import os
from typing import Dict
from collections import Counter

def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(filepath, data):
    base, ext = os.path.splitext(filepath)
    output_filepath = f"{base}R1{ext}"
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def good_prediction(entity: Dict):
    label_list = ["LOC", "ORG", "PER"]
    return entity["label"] in label_list

def self_consistency(entry: Dict):
    entity_list = entry["response_0"] + entry["response_1"] + entry["response_2"] + entry["response_3"] + entry["response_4"]
    if entity_list == []:
        return [], 0
    entity_list = filter(good_prediction, entity_list)
    entity_sc = Counter((entity["entity"], entity["label"]) for entity in entity_list)
    if not entity_sc:
        return [], 0
    sample_sc = sum(entity_sc.values()) / len(entity_sc)
    entity_sc = [{"entity": entity, "label": label, "count": count} for (entity, label), count in entity_sc.items()]
    return entity_sc, sample_sc

if __name__ == "__main__":
    file_path = "en_pud-ud-train-final-4o-iter5.json"
    data = read_json(file_path)
    print(self_consistency(data[0]))
    for entry in data:
        print(entry["id"])
        entity_sc, sample_sc = self_consistency(entry)
        entry["entity_sc"] = entity_sc
        entry["sample_sc"] = sample_sc
    write_json(file_path, data)
