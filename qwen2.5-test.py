from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
cache_dir = "D:/Hugging Face Cache/hub"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",  # Force model to run on GPU
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

prompt = "Extract entities and return them in the JSON format: [{\"entity\": entity1, \"label\": label1}, {\"entity\": entity2, \"label\": label2}, etc.]. Now, extract entities from the following text:\n\"My name is Barack Obama.\"\n"
messages = [
    {"role": "system", "content": "You are an expert Named Entity Recognition (NER) system. Your task is to accept text as input and extract named entities.\nEntities must have one of the following labels: PER (Person), LOC (Location), ORG (Organization)."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
