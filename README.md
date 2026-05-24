# Emotion Recognition from Speech

A deep learning system that listens to a short voice clip and predicts the speaker's emotion — built for the **CodeAlpha Machine Learning Internship (Task 2)**.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/emm-xin19/CodeAlpha-EmotionRecognition_AaminaKhan)
**Notebook:** [`CodeAlpha_EmotionRecognition.ipynb`](./CodeAlpha_EmotionRecognition.ipynb)

---

## Overview

The model classifies an audio clip into one of **8 emotions**:

angry · calm · disgust · fearful · happy · neutral · sad · surprised

Pipeline at a glance: raw audio → resample to 22.05 kHz → trim to a 3-second window → extract 40 MFCCs over 174 frames → normalize with train-set mean/std → CNN inference → softmax over 8 classes.

## Tech Stack

| Layer | Tool |
|---|---|
| Audio processing | `librosa` |
| Model | TensorFlow / Keras (CNN) |
| Training | Google Colab (GPU) |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song. 24 actors (12 male, 12 female) speaking two lexically matched statements across the 8 target emotions, each at two intensity levels.

## Model

A compact CNN trained on MFCC spectrograms.

- Input shape: `(40, 174, 1)`
- ~487K parameters — light enough for free-tier CPU inference
- Trained with class-balanced sampling and a small dropout stack to control overfitting

### Results

| Metric | Value |
|---|---|
| Validation accuracy | ~85% |
| Training accuracy | ~93% |
| Train–val gap | Small (model generalizes, not memorizing) |
| Random baseline | 12.5% (8-way) |

The first iteration overfit hard (train 79% / val 32%). Adding stronger regularization, normalizing MFCCs with train-set statistics, and balancing class sampling closed the gap.

## Repo Layout

```
.
├── CodeAlpha_EmotionRecognition.ipynb   # full training + deployment notebook
├── app.py                               # standalone Gradio app (mirrors HF Space)
├── requirements.txt                     # runtime deps for the Space
├── best_emotion_model.keras             # trained weights
├── label_encoder.pkl                    # sklearn LabelEncoder for class names
└── norm_stats.pkl                       # train mean / std for MFCC normalization
```

## Running Locally

```bash
git clone https://github.com/armadilu/CodeAlpha_EmotionRecognition_AaminaKhan.git
cd CodeAlpha_EmotionRecognition_AaminaKhan
pip install -r requirements.txt
python app.py
```

Opens the Gradio app at `http://127.0.0.1:7860`.

## Reproducing Training

Open `CodeAlpha_EmotionRecognition.ipynb` in Colab, switch the runtime to GPU, and run all cells top-to-bottom. The notebook downloads RAVDESS, extracts MFCCs, trains the CNN, and saves the three artifacts above.

## Acknowledgements

- **RAVDESS** dataset by Livingstone & Russo
- **CodeAlpha** for the internship task and structure

---

Made by **Aamina Khan**
