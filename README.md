# Vowel-Based AI Infant Cry Analysis

**Institution:** SRM Institute of Science and Technology, Kattankulathur  
**Department:** Data Science and Business Systems  
**Course:** B.Tech CSE with specialization in Big Data Analytics  
**Year:** May 2026

---

## Team
| Name | Register Number |
|------|----------------|
| Nimisha Ann Shaji | RA2311027010073 |
| V.N. Preetam | RA2311027010100 |
| K. Bhavya | RA2311027010113 |

**Guide:** Dr. R. Radha, Assistant Professor, DSBS

---

## About the Project
An AI-based system that automatically classifies infant cries into four categories:
- 🍼 **Hungry**
- 😣 **Belly Pain**
- 😴 **Tired**
- 😟 **Discomfort**

The system uses **vowel-based acoustic analysis** combined with MFCC (Mel-Frequency Cepstral Coefficients), spectral, and temporal features. A **Random Forest** classifier (~91% accuracy) outperforms the SVM baseline (~86%).

---

## Tech Stack
- Python (Google Colab)
- librosa, scikit-learn, imbalanced-learn, soundfile
- Dataset: donateacry_corpus

---

## How It Works
1. Audio preprocessing (noise removal, normalization, trimming)
2. Data augmentation (time shift, pitch shift, noise addition)
3. Feature extraction (MFCC + vowel formants F1/F2)
4. Class imbalance handling (SMOTE)
5. Model training (SVM vs Random Forest)
6. Prediction with vowel detection (A/E/I/O/U)

---

## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (Baseline) | ~86% | Low | Low | Imbalanced |
| Random Forest (Final) | ~91% | High | Better | Balanced |

---

## Files
- `main.py` — Full pipeline: preprocessing, feature extraction, training, evaluation
- `Minor_Project_report__Final_.pdf` — Complete project report

---

## SDG Alignment
This project supports **UN SDG Goal 3 — Good Health and Well-being** by helping caregivers better understand infant needs through AI.
