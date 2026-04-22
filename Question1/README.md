# Question1 – NLP Translation Task

## Model Used

Helsinki-NLP/opus-mt-bn-en

## Files

* `translate.py` – Translates Bengali input to English output
* `evaluate.py` – Computes BLEU score using sacrebleu
* `input.txt` – Bengali sentences
* `reference.txt` – Ground truth English sentences
* `output.txt` – Generated translations

## How to Run

```bash
pip3 install --user transformers torch sentencepiece sacrebleu sacremoses numpy
python3 translate.py
python3 evaluate.py
```

## Output

* Translated sentences are saved in `output.txt`
* BLEU score is printed after evaluation

## Note

Model is downloaded automatically from HuggingFace.
Model weights are not included due to size constraints.
