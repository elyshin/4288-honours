import openai
from datetime import datetime
import time
import os
import json
import sys

def gpt_request(message, sentence_index, response_index, model):
    request = {
        "custom_id": str(sentence_index) + "-response_" + str(response_index),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0.5,
            "messages" : message,
            "max_tokens": 300,
            "stop": ['\n\n'],
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
    for result in results:
        sentence_index = results.index(result) // 5
        result = json.loads(result)
        response_index = result["custom_id"].split("-")[1]
        try:
            response = result["response"]["body"]['choices'][0]['message']['content']
            response = response[response.find("["):response.find("]")+1].replace("'", '"')
            sentences[sentence_index][response_index] = json.loads(response)
        except Exception as e:
            print(f"Error processing sentence {sentence_index}, response {response_index}.")            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent = 2, ensure_ascii=False)

def prompt_template(sentence):
    prompt_template = "Extract entities and return them in the JSON format: [{\"entity\": entity1, \"label\": label1}, {\"entity\": entity2, \"label\": label2}, etc.]. If there are no entities, return an empty list: []. Now, extract entities from the following text:\n"
    system_message = "You are an expert Named Entity Recognition (NER) system. Your task is to accept text as input and extract named entities.\nEntities must have one of the following labels: PER (Person), LOC (Location), ORG (Organization)."
    message = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt_template + sentence["text"] + "\nAnswer:"}]
    return message

if __name__ == '__main__':
    file_path = "vi_pud-ud-train.json"
    sentences = read_file(file_path)

    file_request = "vi_pud-ud-train-4o.json"
    with open(file_request, 'w') as file:
        for sentence in sentences:
            current_prompt = prompt_template(sentence)
            for i in range(5):
                if sentence.get("response_"+str(i)) == None:
                    request = gpt_request(current_prompt, sentence["id"], i, "gpt-4o-mini")
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
    result_logs = "vi_train_sc_logs_4o.json"
    with open(result_logs, 'w') as f:
        f.write(client.files.content(result_id).text)
    
    output_path = "vi_pud-ud-train-sc_4o.json"
    file_process(file_path, result_logs, output_path)