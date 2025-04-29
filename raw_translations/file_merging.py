import json


def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    file_order = ["1", "2", "3", "4", "5", "6", "7_1", "7_2", "8_1", "8_2", "9_1", "9_2", "9_3", "10"]
    file_prefix = "vi_pud-ud-test-"
    file_suffix = ".json"
    
    output = []
    id = 1
    for file in file_order:
        file_name = file_prefix + file + file_suffix
        data = read_json(file_name)
        for sentence in data:
            output.append({"id": id, "text": sentence["text"], "entities": sentence["entities"]})
            id += 1

    write_json("vi_pud-ud-test.json", output)