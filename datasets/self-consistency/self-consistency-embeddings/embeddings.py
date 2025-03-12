import openai
from tqdm import tqdm
from datetime import datetime
import time
import os
import json
import sys

def gpt_request(text: str, index, model):
    request = {
        "custom_id": f"{index}",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": model,
            "input": text,
            "encoding_format": "float"
        }
    }
    return request

def read_file(filepath):
    with open(filepath, encoding="utf-8") as file:
        return json.load(file)    

def file_process(original_path, result_logs, output_path):
    sentences = read_file(original_path)
    with open(result_logs) as f:
        results = f.readlines()
    for idx in range(len(sentences)):
        sentences[idx]["embedding"] = json.loads(results[idx])["response"]["body"]["data"][0]["embedding"]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent = 2, ensure_ascii=False)
    
if __name__ == '__main__':
    file_path = "vi_pud-ud-test.json"
    sentences = read_file(file_path)

    file_request = "vi_test_requests.json"
    with open(file_request, 'w') as file:
        for sentence in sentences:
            current_prompt = sentence["text"]
            request = gpt_request(current_prompt, sentence["id"], "text-embedding-ada-002")
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
        endpoint="/v1/embeddings",
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
    result_logs = "vi_embeddings_test_logs.json"
    with open(result_logs, 'w') as f:
        f.write(client.files.content(result_id).text)
    
    output_path = "vi_pud-ud-test-embeddings.json"
    file_process(file_path, result_logs, output_path)