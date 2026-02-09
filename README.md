#  Image Caption Generator using CNN–LSTM (Flickr8k)

This project implements an **Image Caption Generation system** using **Deep Learning**, combining **CNN (VGG16)** for image feature extraction and **LSTM** for natural language generation.  
The model is trained on the **Flickr8k dataset** and generates meaningful captions for unseen images.

---

##  Project Overview

Image captioning is a classic **Computer Vision + NLP** problem where the goal is to generate a natural language description of an image.

###  Key Idea
- **CNN (VGG16)** extracts high-level image features
- **LSTM** learns to generate captions word-by-word
- **Encoder–Decoder architecture** is used

---

##  Model Architecture

### Encoder (Image Feature Extractor)
- Pretrained **VGG16**
- Last fully connected layer output (4096-dim)
- Dropout + Dense layer for feature compression

### Decoder (Caption Generator)
- Word **Embedding layer**
- **LSTM** for sequence modeling
- Dense + Softmax for word prediction

Image → VGG16 → Feature Vector
Caption → Embedding → LSTM
Feature + LSTM → Decoder → Next Word

---

##  Dataset

- **Dataset:** Flickr8k  
- **Images:** 8,000  
- **Captions:** 5 captions per image  

Dataset source (Kaggle):  
https://www.kaggle.com/datasets/adityajn105/flickr8k

---

##  Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- NLTK (BLEU Score)  
- VGG16 (Pretrained CNN)  
- Google Colab / Kaggle Notebook  

---

##  Training Details

- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Batch Size:** 32  
- **Epochs:** 20  
- **Train/Test Split:** 90% / 10%  

A **data generator** is used to avoid RAM overflow during training.

---

##  Evaluation Metric

The model is evaluated using **BLEU scores**:

- **BLEU-1**
- **BLEU-2**

BLEU scores compare generated captions with ground-truth captions to measure linguistic similarity.

---

##  Sample Results

### Example Output:

**Image:**  
`1001773457_577c3a7d70.jpg`

**Predicted Caption:**  
startseq dog running through grass endseq


---

##  Inference Pipeline

1. Load image
2. Resize to 224×224
3. Extract features using VGG16
4. Generate caption word-by-word using trained LSTM
5. Stop when `endseq` token is generated

---

## 📁 Project Structure

├── features.pkl
├── best_model.h5
├── Image_Captioning.ipynb
├── README.md



---

##  How to Run

1. Clone the repository
```bash
git clone https://github.com/Vivekk-007/image_Caption_generator.git
cd image-caption-generator

## Install dependencies

pip install tensorflow nltk tqdm pillow


##  How to Run

Use Attention Mechanism
Replace VGG16 with ResNet / EfficientNet
Train on Flickr30k / MS COCO
Deploy using Hugging Face Spaces
Add Beam Search for better captions


flowchart LR
    A[Input Image 224x224] --> B[VGG16 CNN Encoder]
    B --> C[4096-D Image Feature Vector]
    C --> D[Dense Layer 256]

    E[startseq Caption Input] --> F[Tokenizer]
    F --> G[Embedding Layer]
    G --> H[LSTM]

    D --> I[Feature Fusion (Add)]
    H --> I

    I --> J[Dense Layer]
    J --> K[Softmax Output]
    K --> L[Predicted Word]
+------------------+
|   Input Image    |
|   (224 x 224)    |
+------------------+
          |
          ▼
+------------------+
|  VGG16 Encoder   |
|  (Pretrained)   |
+------------------+
          |
          ▼
+------------------+
|  Image Features  |
|   (4096-dim)    |
+------------------+
          |
          ▼
+------------------+           +--------------------+
| Dense (256)      |           | Text Input         |
+------------------+           | (startseq)         |
          |                     +--------------------+
          |                               |
          |                               ▼
          |                     +--------------------+
          |                     | Embedding Layer    |
          |                     +--------------------+
          |                               |
          |                               ▼
          |                     +--------------------+
          |                     | LSTM Decoder       |
          |                     +--------------------+
          |                               |
          +------------+------------------+
                       ▼
               +--------------------+
               | Feature Fusion     |
               | (Add Operation)    |
               +--------------------+
                       |
                       ▼
               +--------------------+
               | Dense + Softmax    |
               +--------------------+
                       |
                       ▼
               Predicted Caption






