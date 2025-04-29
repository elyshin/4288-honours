from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import tqdm
import json
from typing import List, Dict
import os
import argparse
from filterretrieval import *
import random

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
cache_dir = "D:/Hugging Face Cache/hub"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",  # Force model to run on GPU
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(filepath, data, filter_type, retrieval_method):
    base, ext = os.path.splitext(filepath)
    output_filepath = f"{base}-{filter_type}-{retrieval_method}{ext}"
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

def prompt_template(sentence, examples: List[Dict], num_examples):
    # Function to remove 'count' from entity_sc entries
    def clean_entity_sc(entity_sc):
        return [
            {key: value for key, value in entry.items() if key != "count"}
            for entry in entity_sc
        ]
    
    cleaned_entities = [clean_entity_sc(example['entity_sc']) for example in examples]
    
    prompt_template = "Extract entities and return them in the JSON format: [{\"entity\": entity1, \"label\": label1}, {\"entity\": entity2, \"label\": label2}, etc.]. If there are no entities, return an empty list: []. Now, extract entities from the following text:\n"
    system_message = "You are an expert Named Entity Recognition (NER) system. Your task is to accept text as input and extract named entities.\nEntities must have one of the following labels: PER (Person), LOC (Location), ORG (Organization)."
    
    current_prompt = prompt_template
    for i in range(num_examples):
        current_prompt += f"Text: {examples[i]['text']}\nAnswer: {cleaned_entities[i]}\n\n"
    
    current_prompt += f"Text: {sentence['text']}\nAnswer:"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": current_prompt}  
    ]
    print(messages)
    return messages

def generation(prompt, max_new_tokens=512, temperature=0):
    prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature = temperature
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def parser():
    parser = argparse.ArgumentParser(description="Run the Qwen model on a dataset with custom filtering and retrieval strategies.")
    parser.add_argument("test_path", type=str, help="Path to the test dataset.")
    parser.add_argument("train_path", type=str, help="Path to the training dataset.")
    parser.add_argument("filter_type", type=str, default="two_filtering", help="Filtering strategy to use. Options: two_filtering, entity_filtering, sample_filtering, none.")
    parser.add_argument("entity_threshold", type=int, default=3, help="Entity threshold for filtering.")
    parser.add_argument("sample_threshold", type=int, default=3, help="Sample threshold for filtering.")
    parser.add_argument("retrieval_method", type=str, default="diverse_sc", help="Retrieval method to use. Options: random, nearest, diverse_nearest, diverse_sc.")
    parser.add_argument("k", type=int, default=5, help="Number of examples to retrieve.")
    return parser.parse_args()

def filtered_dataset(dataset, filter_type, entity_threshold, sample_threshold):
    if filter_type == "two_filtering":
        return two_filtering(dataset, entity_threshold, sample_threshold)
    elif filter_type == "entity_filtering":
        return entity_filtering(dataset, entity_threshold)
    elif filter_type == "sample_filtering":
        return sample_filtering(dataset, sample_threshold)
    elif filter_type == "none":
        return dataset
    else:
        raise ValueError("Invalid filter type. Options: two_filtering, entity_filtering, sample_filtering.")

def retrieval(sentence, dataset, retrieval_method, k):
    retrieved_dataset = knn_retrieval(sentence, dataset, 30)
    if retrieval_method == "random":
        return random.sample(dataset, k)
    elif retrieval_method == "nearest":
        return retrieved_dataset[:k]
    elif retrieval_method == "diverse_nearest":
        return random.sample(retrieved_dataset, k)
    elif retrieval_method == "diverse_sc":
        return sorted(retrieved_dataset, key=lambda x: x["sample_sc"], reverse=True)[:k]
    else:
        raise ValueError("Invalid retrieval method. Options: random, nearest, diverse_nearest, diverse_sc.")

def main():
    args = parser()
    set_seed(42)
    random.seed(42)
    test_sentences = read_json(args.test_path)
    train_sentences = read_json(args.train_path)
    filtered_train = filtered_dataset(train_sentences, args.filter_type, args.entity_threshold, args.sample_threshold)
    test_sentences = random.sample(test_sentences, 300)
    total_TP, total_FP, total_FN = 0, 0, 0
    for sentence in tqdm.tqdm(test_sentences):
        examples = retrieval(sentence, filtered_train, args.retrieval_method, args.k)
        messages = prompt_template(sentence, examples, args.k)
        retries = 0
        max_retries = 5

        while retries < max_retries:
            response = generation(messages, 512, 0.5)
            response = response[response.find("["):response.find("]")+1].replace("'", '"')
            
            try:
                parsed_response = json.loads(response)
                break
            except Exception as e:
                print(response)
                print(f"Error at sentence {sentence['id']}, retrying... ({retries+1}/{max_retries})")
                retries += 1
    
        if retries == max_retries:
            print(f"Max retries reached for sentence {sentence['id']}, moving on...")
            parsed_response = []
        sentence.update({"response": parsed_response})
        sentence.pop("embedding", None)
        TP, FP, FN = calculate_ner_metrics(sentence["entities"], parsed_response)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision, recall, TPR, f1 = metrics(total_TP, total_FP, total_FN)
    print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}, Precision: {precision:.2f}, Recall: {recall:.2f}, TPR: {TPR:.2f}, F1: {f1:.2f}")

    # Add metrics and write results to a new JSON file
    total_metrics = {
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "precision": precision,
        "recall": recall,
        "TPR": TPR,
        "f1": f1
    }
    test_sentences.insert(0, total_metrics)
    write_json(args.test_path, test_sentences, args.filter_type, args.retrieval_method)


if __name__ == "__main__": 
    # Read the vi_pud-ud-test-embeddings-none-nearest.json dataset, calculate precision, recall, TPR, F1 of all sentences and readd the total_metrics
    main()

    # Commander to enter: python3 qwen-sc.py vi_pud-ud-test-embeddings.json vi_pud-ud-sc-final.json none 3 3 diverse_nearest 5