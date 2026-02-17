from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

login()

model_path = "distilbert-reviews-genres"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.push_to_hub("Sushantak17/distilbert-review-genres")
tokenizer.push_to_hub("Sushantak17/distilbert-review-genres")

print("Upload completed successfully")
