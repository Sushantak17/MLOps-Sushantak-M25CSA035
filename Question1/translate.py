from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-bn-en"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

with open("input.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

translations = []

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    translations.append(output)

with open("output.txt", "w", encoding="utf-8") as f:
    for line in translations:
        f.write(line + "\n")

print("First translated sentence:")
print(translations[0])