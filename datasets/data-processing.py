import os
import json

def read_iob2_dataset(filepath, marker):
    with open(filepath, 'r', encoding='utf-8') as file:
        sentences = []
        id = 0
        current_entity = False
        for line in file:
            # extract full sentence
            if line.startswith('# text ='):
                id += 1
                text = line.strip().split('=')[1].strip()
                entities = []

            # extract tokens and labels
            if 'B-' in line and marker in line:
                if current_entity:
                    print(current_entity)
                    entities.append({"entity": current_entity, "label": label})
                    current_entity = ""
                parts = line.strip().split('\t')
                current_entity = parts[1]
                label = parts[2][2:]

            if 'I-' in line and marker in line:
                parts = line.strip().split('\t')
                print(parts[1])
                current_entity += " " + parts[1]
            
            if line == '\n':
                if current_entity != "":
                    entities.append({"entity": current_entity, "label": label})
                    current_entity = ""
                sentence = {"id": id, "text": text, "entities":  entities}
                sentences.append(sentence)
    return sentences

def write_file(sentences, output_path):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, ensure_ascii=False, indent=2)
            

if __name__ == "__main__":
    file_path = "datasets/zh_pud-ud-test.iob2"
    sentences = read_iob2_dataset(file_path, "linpq")
    write_file(sentences, "datasets/zh_pud-ud-test.json")

