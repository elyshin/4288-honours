from datetime import datetime
import json
from typing import List, Dict
import os
import argparse
import random
import openai
import sys
import time
import re

def gpt_request(message, sentence_index, model):
    request = {
        "custom_id": str(sentence_index),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0.0,
            "messages" : message,
            "max_tokens": 300,
            "stop": ['\n\n'],
        }
    }
    return request


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
    '''for i in range(num_examples):
        current_prompt += f"Text: {examples[i]['text']}\nAnswer: {cleaned_entities[i]}\n\n"'''
    
    current_prompt += f"Text: {sentence['text']}\nAnswer:"
    
    messages = [
        {"role": "user", "content": system_message + current_prompt}  
    ]
    
    return messages


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

def file_process(test_dataset, result_logs):
    total_TP, total_FP, total_FN = 0, 0, 0
    with open(result_logs) as f:
        results = f.readlines()
    for idx in range(len(test_dataset)):
        response = json.loads(results[idx])["response"]["body"]['choices'][0]['message']['content']
        response = response[response.find("["):response.find("]")+1].replace("'", '"')
        try:
            parsed_response = json.loads(response)
            test_dataset[idx].update({"response": parsed_response})
        except Exception as e:
            print(response)
            print(f"Error at sentence {test_dataset[idx]['id']}")
            continue
        TP, FP, FN = calculate_ner_metrics(test_dataset[idx]["entities"], parsed_response)
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
    test_dataset.insert(0, total_metrics)

def main():
    args = parser()
    random.seed(42)
    test_sentences = read_json(args.test_path)
    train_sentences = read_json(args.train_path)

    file_request = args.train_path.replace(".json", "-logs.json")
    # Create batch file
    with open(file_request, 'w') as file:
        for sentence in test_sentences:
            current_prompt = prompt_template(sentence, [], args.k)
            request = gpt_request(current_prompt, sentence["id"], "gpt-4.1-mini")
            file.write(json.dumps(request) + "\n")

    # uploading batch input file to openai
    openai.api_key = os.getenv('OPENAPI_API_KEY')

    client = openai.OpenAI()
    batch_input_file = client.files.create(
        file=open(file_request, "rb"),
        purpose="batch"
    )

    #creating batch job
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    #print file batch_id
    print("\n", f"Batch ID: {batch_job.id}\n")

    #check status of batch every 5 minutes
    completion = False
    while not completion: 
        status = client.batches.retrieve(batch_job.id)
        timestamp = datetime.now().strftime('%m-%d_%H-%M')
        print(timestamp, "\n", status, "\n")
        if status.status == "completed": 
            completion = True
            print("Batch completed. Calculating results.")
        elif status.status in ["canceled", "failed"]: 
            sys.exit("Batch has failed or been canceled. Please try again.")
        else:
            print("Batch in progress. Next query in 1 minute.")
            time.sleep(60)
    
    #retrieve results
    result_id = client.batches.retrieve(batch_job.id).output_file_id

    #results_logs for raw data check
    result_logs = args.train_path.replace(".json", f"-logs-{args.filter_type}-{args.retrieval_method}.json")
    with open(result_logs, 'w') as f:
        f.write(client.files.content(result_id).text)

    file_process(test_sentences, result_logs)
    write_json(args.test_path, test_sentences, args.filter_type, args.retrieval_method)

if __name__ == "__main__": 
    main()
    # Commander to enter: python 4o-sc.py en_pud-ud-test-embeddings.json en_pud-ud-train-final-4o.json two_filtering 3 3 diverse_sc 5