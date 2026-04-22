import sacrebleu

with open("output.txt", "r", encoding="utf-8") as f:
    preds = [line.strip() for line in f if line.strip()]

with open("reference.txt", "r", encoding="utf-8") as f:
    refs = [line.strip() for line in f if line.strip()]

bleu = sacrebleu.corpus_bleu(preds, [refs])

print("BLEU Score:", bleu.score)