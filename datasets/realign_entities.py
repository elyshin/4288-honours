import json
import os

def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(filepath, data):
    base, ext = os.path.splitext(filepath)
    output_filepath = f"{base}-results{ext}"
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def realign_entities(vietnamese_file, english_file):
    vietnamese_data = read_json(vietnamese_file)
    english_data = read_json(english_file)

    for vi_sentence, en_sentence in zip(vietnamese_data, english_data):
        en_sentence['entities'] = vi_sentence.get('entities', [])

    write_json(english_file, english_data)

if __name__ == "__main__":
    vietnamese_file = "/d:/NUS/DSA4288/Project/datasets/raw datasets/vi_pud-ud-test.json"
    english_file = "/d:/NUS/DSA4288/Project/datasets/en_pud-ud-test.json"
    realign_entities(vietnamese_file, english_file)
