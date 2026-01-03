# Project Report: Propaganda Detection in Arabic Narratives (Idea 6)

**Course:** AIS411 - Natural Language Processing (NLP)
**Date:** January 3, 2026
**Status:** ✅ Completed (All Phases + Demo)

---

## 1. Executive Summary
This project aims to detect propaganda techniques in Arabic news and social media posts regarding the war on Gaza (2023–2025). Using the **FigNews 2024** dataset, we implemented a complete NLP pipeline comprising data preprocessing, visualization, and a comparative study between Classical Machine Learning and Deep Learning models.

The study demonstrates that while deep learning models (**AraBERT**) initially struggle with imbalanced data, optimizing them with **Cost-Sensitive Learning (Weighted Loss)** allows them to outperform robust statistical baselines (**Logistic Regression**), achieving a final F1-Macro score of **0.545**.

---

## 2. Phase 1: Data Analysis & Preprocessing

### 2.1 Objective
To acquire, clean, and explore a dataset of Arabic narratives to prepare it for binary classification (Propaganda vs. Non-Propaganda).

### 2.2 Dataset Description
* **Source:** FigNews 2024 Shared Task (CAMeL Lab).
* **Content:** Arabic posts related to the Israeli War on Gaza.
* **Size:** 6,342 samples.
* **Labeling:** A "Majority Vote" logic was applied to annotator data to create binary labels.

### 2.3 Preprocessing Pipeline
We implemented a strict cleaning function to ensure high-quality input for the models:
1.  **Noise Removal:** Stripped URLs, English characters, numbers, and emojis.
2.  **Encoding Fix:** Removed invisible Unicode control characters (Right-to-Left Isolates) that were causing visualization errors.
3.  **Normalization:** Standardized whitespace.

### 2.4 Exploratory Data Analysis (EDA)
* **Class Imbalance:** The dataset is imbalanced with a ratio of approximately 2:1.
    * *Propaganda:* 4,150 samples (65.4%)
    * *Non-Propaganda:* 2,192 samples (34.6%)
* **Visualization:** A word cloud of the Propaganda class highlighted frequent terms such as "Gaza" (غزة), "Hamas" (حماس), and "The Occupation" (الاحتلال).

---

## 3. Phase 2: Comparative Study & Implementation

### 3.1 Objective
To implement and compare two distinct classification models, satisfying the course requirement for model comparison.

### 3.2 Models

**Model A: Deep Learning (AraBERT)**
* **Architecture:** `aubmindlab/bert-base-arabertv02` (Transformer).
* **Configuration:** Fine-tuned using Hugging Face Transformers with Mixed Precision (FP16) on an NVIDIA RTX 3050.

**Model B: Baseline (Classical ML)**
* **Architecture:** Logistic Regression.
* **Features:** TF-IDF (Top 5,000 features).
* **Configuration:** `class_weight='balanced'` was used to handle imbalance.

### 3.3 Initial Results
| Model | Accuracy | F1 Macro | Observation |
| :--- | :--- | :--- | :--- |
| **AraBERT (Base)** | **65.3%** | 0.508 | High accuracy but "lazy" (biased toward majority class). |
| **Baseline (LogReg)** | 54.5% | **0.526** | **Winner.** Better handling of the minority class. |

---

## 4. Phase 3: Model Optimization

### 4.1 Problem Identification
In Phase 2, AraBERT achieved high accuracy by predicting "Propaganda" for almost every sample. This resulted in a poor F1 score, as it failed to detect "Non-Propaganda" posts effectively.

### 4.2 Optimization Strategy
We implemented **Cost-Sensitive Learning** using a **Weighted Cross-Entropy Loss** function.
* **Mechanism:** The model was penalized more heavily for mistakes on the minority class.
* **Weights:**
    * Non-Propaganda: `1.45` (High Penalty)
    * Propaganda: `0.76` (Lower Penalty)

### 4.3 Final Results
| Metric | Baseline | **AraBERT (Optimized)** | Improvement |
| :--- | :--- | :--- | :--- |
| **F1 Macro** | 0.526 | **0.545** | **+3.6%** |
| **Accuracy** | 54.5% | 58.6% | +7.5% |

**Conclusion:** The optimized AraBERT model successfully beat the baseline. Although overall accuracy dropped compared to the "lazy" version, the model is now "fairer" and significantly better at distinguishing between the two classes.

---

## 5. Deployment (Bonus)
An interactive web interface was developed using **Gradio** to demonstrate the model in real-time.
* **Input:** Users can type any Arabic text.
* **Output:** The system returns a probability score for "Propaganda" vs. "Non-Propaganda."
* **Test Case:** The input *"The brutal Zionist enemy commits savage massacres"* was correctly flagged as **Propaganda (56%)**, validating the model's ability to detect emotionally loaded language.

---

## 6. Tools & Technologies
* **Language:** Python 3.11
* **Libraries:** PyTorch, Transformers (Hugging Face), Scikit-Learn, Pandas, Arabic-Reshaper.
* **Hardware:** NVIDIA RTX 3050 (6GB VRAM) with CUDA 12.1.