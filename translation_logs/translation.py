import openai
from tqdm import tqdm
from datetime import datetime
import time
import os
import json
import sys

prompt = """Translate the following sentence into Vietnamese, ensuring named entities (people, organizations, locations, dates) are translated correctly. Keep the entities aligned with their correct translations. Provide the result in JSON format with "text" for the translated sentence and "entities" for the list of identified entities.

### Input:
""" 

def build_prompt(sentence: str):
    sentence_str = json.dumps(sentence)
    return prompt + sentence_str

def gpt_request(prompt: str, index, model):
    request = {
        "custom_id": f"request-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0.0,
            "messages" : [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "stop": ['\n\n'],
        }
    }
    return request

def read_file(filepath):
    with open(filepath, encoding="utf-8") as file:
        return json.load(file)
    
def file_process(result_logs, output_path):
    with open(result_logs) as f:
        results = f.readlines()
    output = []
    id = 891
    for line in results:
        sentence = json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]
        sentence = json.loads(sentence.strip("`").strip("json"))
        sentence["id"] = id
        id += 1
        output.append(sentence)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent = 2, ensure_ascii=False)
    

if __name__ == '__main__':
    file_path = "datasets/en_pud-ud-test.json"
    sentences = read_file(file_path)

    file_request = "datasets/batch_requests.json"
    with open(file_request, 'w') as file:
        for sentence in sentences[890:]:
            current_prompt = build_prompt(sentence)
            request = gpt_request(current_prompt, sentence["id"], "gpt-4o")
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
    result_logs = "datasets/result_logs_10.json"
    with open(result_logs, 'w') as f:
        f.write(client.files.content(result_id).text)

    output_path = "datasets/vi_pud-ud-test-10.json"
    file_process(result_logs, output_path)