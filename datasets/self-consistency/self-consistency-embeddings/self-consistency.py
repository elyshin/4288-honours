from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import tqdm
import json
from typing import List, Dict
from collections import Counter
import os

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
cache_dir = "Hugging Face Cache/hub"
print("Is CUDA available?", torch.cuda.is_available())

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",  # Force model to run on GPU
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

prompt_template = "Extract entities and return them in the JSON format: [{\"entity\": entity1, \"label\": label1}, {\"entity\": entity2, \"label\": label2}, etc.]. If there are no entities, return an empty list: []. Now, extract entities from the following text:\n"
system_message = "You are an expert Named Entity Recognition (NER) system. Your task is to accept text as input and extract named entities.\nEntities must have one of the following labels: PER (Person), LOC (Location), ORG (Organization).\n"

def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(filepath, data):
    base, ext = os.path.splitext(filepath)
    output_filepath = f"{base}-sc{ext}"
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def calculate_ner_metrics(true_labels: List[Dict], pred_labels: List[Dict]):
    try:
        true_set = {(item["entity"].lower(), item["label"].lower()) for item in true_labels}
        pred_set = {(item["entity"].lower(), item["label"].lower()) for item in pred_labels}
    except Exception as e:
        return 0, 0, 0

    TP = len(true_set & pred_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set) 

    return TP, FP, FN
    
def metrics(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, TPR, f1  

if __name__ == "__main__": 
    set_seed(42)  # Set seed for replicability
    filepath = "en_pud-ud-train.json"
    sentences = read_json(filepath)
    total_TP, total_FP, total_FN = 0, 0, 0
    for sentence in tqdm.tqdm(sentences):
        text = sentence["text"]
        current_prompt = prompt_template + text + "\n" 
        messages = [
            {"role": "user", "content": system_message + current_prompt}  
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        for index in range(5):
            tag = "response_" + str(index)
            if sentence.get(tag):
                continue
            
            while True:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature = 0.5,
                    top_k = 40
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                #parsing response
                response = response[response.find("["):response.find("]")+1]
                try:
                    parsed_response = json.loads(response)
                    sentence.update({tag: parsed_response})
                    break
                except Exception as e:
                    print(f"Error at sentence {sentence['id']}, response #{tag[-1]}, retrying...")
    write_json(filepath, sentences)
        