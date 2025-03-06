from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
import torch
import tqdm
import json
from typing import List, Dict
from collections import Counter

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
cache_dir = "D:/Hugging Face Cache/hub"
print("Is CUDA available?", torch.cuda.is_available())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",  # Force model to run on GPU
    cache_dir=cache_dir,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

prompt_template = "You are an expert Named Entity Recognition (NER) system. Your task is to accept text as input and extract named entities.\nEntities must have one of the following labels: PER (Person), LOC (Location), ORG (Organization). Extract entities and return them in the JSON format: [{\"entity\": entity1, \"label\": label1}, {\"entity\": entity2, \"label\": label2}, etc.]. Now, extract entities from the following text:\n"

def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_ner_metrics(true_labels: List[Dict], pred_labels: List[Dict]):
    # Convert lists into (entity, label) tuples for proper counting
    true_set = {(item["entity"], item["label"]) for item in true_labels}
    print(true_set)
    pred_set = {(item["entity"], item["label"]) for item in pred_labels}
    print(pred_set)

    # Count True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(true_set & pred_set)  # Correctly predicted entities
    FP = len(pred_set - true_set)  # Wrong predictions
    FN = len(true_set - pred_set)  # Missed entities

    return TP, FP, FN
    
def metrics(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, TPR, f1  

if __name__ == "__main__": 
    set_seed(42)  # Set seed for replicability
    filepath = "datasets/en_pud-ud-test.json"
    sentences = read_json(filepath)
    total_TP, total_FP, total_FN = 0, 0, 0
    for sentence in tqdm.tqdm(sentences[:1]):
        text = sentence["text"]
        current_prompt = prompt_template + text + "\n" + "<think>\n"
        messages = [
            {"role": "user", "content": current_prompt}  # Use current_prompt here
        ]
        print(messages)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature = 0.6
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        sentence.update({"response": response})
        #cutting response between two square brackets but including them
        response = response[response.find("["):response.find("]")+1]
        print(response)
        print(sentence["entities"])
        try:
            parsed_response = json.loads(response)
        except Exception as e:
            continue
        
        TP, FP, FN = calculate_ner_metrics(sentence["entities"], parsed_response)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision, recall, TPR, f1 = metrics(total_TP, total_FP, total_FN)
    print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}, Precision: {precision:.2f}, Recall: {recall:.2f}, TPR: {TPR:.2f}, F1: {f1:.2f}")