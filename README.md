# Assignment 4: Transformer Optimization using Ray Tune & Optuna

**Name:** Sushantak Parashar Jha  
**Roll No:** M25CSA035  
**Course:** ML-DL-Ops  

---

## Overview
This project focuses on optimizing a Transformer-based English-to-Hindi translation model using **Ray Tune** and **OptunaSearch**. The objective was to match or exceed the baseline BLEU score while significantly reducing training time and computational cost.

---

## Results Summary
- **Baseline BLEU Score:** 83.69 (100 epochs, ~104 min)  
- **Tuned Model BLEU Score:** 83.71 (12 epochs, ~13.6 min)  
- **Speed Improvement:** ~8.3× faster convergence  
- **Training Time Reduced:** ~87%  

---

## Key Improvements
- Higher learning rate enabled faster convergence  
- Reduced model size (d_model = 256, 3 layers) maintained performance  
- OptunaSearch + ASHA enabled efficient hyperparameter tuning  
- Significant reduction in training cost without loss in accuracy  

---

## Repository Contents
- `M25CSA035_ass_4_tuned_en_to_hi.ipynb` → Tuned training notebook  
- `M25CSA035_ass_4_report.pdf` → Detailed report  
- `M25CSA035_ass_4_best_model.pth` → Best model weights  
- `tune_train.py` → Training script (Ray Tune)  

---

## Conclusion
The tuned model achieved comparable translation quality to the baseline while using a fraction of the computational resources, demonstrating the effectiveness of intelligent hyperparameter optimization.