# Natural Language Processing Homeworks - HW2

This repository contains solutions for Homework 2 of the Natural Language Processing (NLP) course, focusing on **Part-of-Speech (PoS) Tagging** and **Named Entity Recognition (NER)** using deep learning and spaCy.

---

## 📁 Project Structure

```
code/
├── PoS Tagging/
│   ├── main.ipynb         # Jupyter notebook for PoS tagging (PyTorch)
│   ├── models/
│   │   └── POSTagger.py   # PyTorch model definition for PoS tagging
│   └── datasets/
│       ├── train.json     # Training data for PoS tagging
│       └── test.json      # Test data for PoS tagging
├── Name Entity Recognition/
│   ├── main.ipynb         # Jupyter notebook for NER (spaCy)
│   └── datasets/
│       └── Entity Recognition in Resumes.json  # NER dataset
```

---

## 📚 Tasks

### 1. Part-of-Speech Tagging

- **Framework:** PyTorch
- **Description:** Sequence labeling task to assign PoS tags to each word in a sentence.
- **Features:**
  - Data preprocessing and vocabulary building
  - Model: RNN-based tagger (custom `POSTagger`)
  - Training and evaluation with accuracy, precision, recall, and F1-score
  - Visualization: Loss curve, per-class metrics, confusion matrix, and class frequency plots

### 2. Named Entity Recognition

- **Framework:** spaCy
- **Description:** Extract named entities (e.g., names, skills, locations) from resume texts.
- **Features:**
  - Data cleaning and conversion to spaCy format
  - Custom entity classes: `COLLEGE NAME`, `LOCATION`, `DESIGNATION`, `EMAIL ADDRESS`, `NAME`, `SKILLS`
  - Model: spaCy NER pipeline (trained from scratch)
  - Evaluation: Entity-level and token-level precision, recall, F1-score
  - Visualization: Training loss curve

---

## 🚀 How to Run

1. **Clone the repository**
   ```sh
   git clone https://github.com/amirrezabsh/POSTagging-NER.git
   cd POSTagging-NER/code
   ```

2. **Install dependencies**
   - For PoS Tagging: See the first cell in `PoS Tagging/main.ipynb`
   - For NER: See the first cell in `Name Entity Recognition/main.ipynb`

3. **Run the notebooks**
   - Open each notebook in VS Code or Jupyter and run all cells.

---

## 📊 Results

- **PoS Tagging:** Reports accuracy, precision, recall, F1-score per class, and visualizes confusion matrix and class distributions.
- **NER:** Reports entity-level and token-level metrics, with tabulated results for each entity type.

---

## 📝 Notes

- All code is written for educational purposes and may require adaptation for other datasets.
- Datasets are expected to be in the `datasets/` folders as shown above.
- For questions or issues, please open an issue or contact the repository owner.

---

## 👩‍💻 Authors

- Amirreza Behmanesh
- MSc NLP, Third Semester

---

## 📄 License

This project is for academic use only.