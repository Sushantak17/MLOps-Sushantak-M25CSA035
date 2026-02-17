from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Sushantak17/distilbert-review-genres"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("Model loaded successfully from Hugging Face")
