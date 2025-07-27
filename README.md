# 🤖 DeepToxicDetector – Toxic Comment Classifier Using Deep Learning

This project focuses on detecting **toxic language** in user-generated content (e.g., YouTube, Reddit) using **deep learning models**. The classifier labels text across multiple categories like `toxic`, `obscene`, `insult`, `threat`, and more. It is trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset.

---

## 📌 Features

- 📚 **Multi-label Classification**: Each comment can have multiple toxicity labels
- 🧠 **Deep Learning Models**: LSTM, Bi-LSTM, and CNN architectures implemented
- 🗣️ **Word Embeddings**: Pre-trained GloVe embeddings used for semantic understanding
- 📊 **Evaluation**: Precision, Recall, F1-Score across all six labels
- 🧪 **Experimental Setup**: Dropout tuning, Batch Size experimentation
- 📥 **CLI-based Testing**: Interactive script to test sample comments (`toxicom.py`)

---

## 🔍 Toxicity Categories

| Label            | Description                         |
|------------------|-------------------------------------|
| `toxic`          | General toxicity                    |
| `severe_toxic`   | Extremely rude or hateful           |
| `obscene`        | Offensive or lewd language          |
| `threat`         | Threatening violence                |
| `insult`         | Personal insults or attacks         |
| `identity_hate`  | Hate speech targeting identity      |

---


## ⚙️ Training Configuration

- **Tokenizer**: Keras Tokenizer
- **Embeddings**: 100D GloVe word vectors
- **Vocabulary Size**: 50,000 words
- **Max Sequence Length**: 200 tokens
- **Model Architecture**: Bidirectional LSTM with 2 hidden layers and dropout
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam optimizer
- **Batch Size**: 128
- **Epochs**: 10

---

## 📊 Model Performance

### F1-Scores by Category

| Category       | F1-Score | Precision | Recall |
|----------------|----------|-----------|--------|
| Toxic          | 0.89     | 0.91      | 0.87   |
| Obscene        | 0.88     | 0.90      | 0.86   |
| Insult         | 0.85     | 0.87      | 0.83   |
| Severe Toxic   | 0.74     | 0.76      | 0.72   |
| Identity Hate  | 0.72     | 0.74      | 0.70   |
| Threat         | 0.68     | 0.71      | 0.65   |

### Overall Metrics
- **Average F1-Score**: 0.79
- **Training Accuracy**: 95.2%
- **Validation Accuracy**: 92.8%

---



## 📈 Model Architecture

```
Input Layer (200,)
    ↓
Embedding Layer (50000, 100)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (6 units, sigmoid activation)
```

---

## 🧪 Experimental Results

### Hyperparameter Tuning
- **Dropout Rate**: Tested 0.2, 0.3, 0.5 → Best: 0.3
- **LSTM Units**: Tested 32, 64, 128 → Best: 64
- **Batch Size**: Tested 64, 128, 256 → Best: 128
- **Learning Rate**: Tested 0.001, 0.01 → Best: 0.001

### Data Distribution
- **Total Comments**: 159,571
- **Toxic Comments**: 15,294 (9.6%)
- **Clean Comments**: 144,277 (90.4%)

---

## 🔮 Future Enhancements

- 🚀 **Web Deployment**: Deploy as a Streamlit or Flask web application
- 🤖 **Advanced Models**: Fine-tune with BERT, RoBERTa, or DistilBERT
- 🌍 **Multilingual Support**: Handle code-mixed and multilingual comments
- ⚡ **Real-time Processing**: Apply to live chat moderation systems
- 📱 **API Integration**: RESTful API for third-party applications
- 🔍 **Explainability**: Add LIME/SHAP for model interpretability

---

## 📋 Requirements

```
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
nltk>=3.7
```

---

## 📝 Dataset Information

The model is trained on the **Jigsaw Toxic Comment Classification Challenge** dataset, which contains:
- **159,571** Wikipedia comments
- **6 toxicity labels** (multi-label classification)
- **Human-annotated** ground truth labels
- **Imbalanced dataset** with class weighting applied

---


## 📄 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software with proper attribution.

---

## 🙋‍♂️ About the Author

@vignshh7

**Connect with me:**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🐱 GitHub: [Your GitHub Profile]

---

## 🙏 Acknowledgments

- **Jigsaw/Conversation AI** for providing the dataset
- **Stanford NLP Group** for GloVe word embeddings
- **Kaggle Community** for insights and discussions
- **TensorFlow/Keras** team for the deep learning framework

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star! ⭐

---

*Feel free to explore, fork, and reuse parts of this repository for learning and research purposes.*
