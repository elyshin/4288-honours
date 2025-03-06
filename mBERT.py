import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

cache_dir = "D:/Hugging Face Cache/hub"
# Load mBERT tokenizer and model for NER
model_name = "google-bert/bert-base-multilingual-cased"  # Fine-tuned mBERT for NER
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)

# Example sentence in multiple languages (English, German, Swedish)
sentences = [
    "Barack Obama was born in Hawaii.",  # English
    "Angela Merkel wurde in Hamburg geboren.",  # German
    "Greta Thunberg är från Sverige."  # Swedish
]

# Load NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Run NER
for sentence in sentences:
    print(f"\nSentence: {sentence}")
    ner_results = ner_pipeline(sentence)
    for entity in ner_results:
        print(f"{entity['word']} -> {entity['entity']} (score: {entity['score']:.3f})")