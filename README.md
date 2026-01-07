# Satellite Imagery–Based Property Valuation

This project builds a **multimodal property valuation system** that combines structured housing data with satellite imagery to predict real estate prices.
The approach integrates traditional tabular modeling with deep visual feature extraction and residual learning to improve generalization.

---

## 📁 Repository Structure

├── data_fetcher.py
├── preprocessing.ipynb
├── model_training.ipynb
└── README.md


---

## 📊 Dataset

- The tabular dataset (CSV) was **provided as part of the problem statement**.
- It contains property-level attributes such as:
  - Location (latitude, longitude)
  - Structural features (size, rooms, year built, etc.)
  - Sale price (target variable)

---

## Satellite Image Acquisition

Satellite images are programmatically downloaded using **Esri World Imagery** tiles.

### Key details:
- **Zoom level:** 19
- **Image size:** ~256×256 (later resized for CNN input)
- **Naming convention:** `{row_index}.jpg`
- **Storage directory for training images:** `sat_images/`
- **Storage directory for test images:** `sat1_images/`

### Script used
`data_fetcher.py`

### How to run
```bash
python data_fetcher.py
```

## Preprocessing & Feature Engineering

**Notebook:** `preprocessing.ipynb`

This notebook handles:

- Data cleaning and missing-value handling  
- Date-based feature extraction  
- Domain-driven feature engineering  
- Geographic clustering  
- Image preprocessing  
- Grad-CAM analysis for visual interpretability  
- Preparation of tabular features and image embeddings  

---

## Model Training

**Notebook:** `model_training.ipynb`

This notebook contains experiments with multiple modeling strategies:

- Tabular-only XGBoost baseline  
- Early fusion (tabular + image embeddings)  
- Wealth-aware multimodal modeling  
- **Residual multimodal architecture (final model)**  

### Final Architecture (High Level)

- **Base model:** XGBoost trained on tabular features  
- **Residual model:** XGBoost trained on image embeddings to predict tabular model errors  

**Final prediction:**

Final Price = Tabular Prediction + Image-Based Residual Correction

This design ensures that satellite imagery contributes only where tabular features fail, improving robustness and generalization.

---

## 📈 Evaluation & Prediction

- Models are evaluated using **RMSE** and **R²** on a held-out test set.
- With best model gave **RMSE** - 101,652 and **R²** - 0.9148
  
The final notebook also includes code to:
- Generate predictions on **unlabeled test data**
- Export predictions in **CSV format** for evaluation

---

## Environment & Dependencies

- **Python:** 3.12.4  
  (Other recent Python 3 versions should also work)

**Core libraries:**
- NumPy  
- Pandas  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras  
- Pillow  
- Requests  

A GPU is **not required**, but may speed up image embedding generation.

---

## 🔁 Reproducibility Notes

- Image filenames are aligned with dataset row indices to ensure correct mapping  
- All preprocessing steps are deterministic  
- CNN weights are frozen to avoid leakage and instability  
- Random seeds are fixed where applicable  

---

## Notes

This repository is structured for **clarity, reproducibility, and academic evaluation**, with a clear separation between data acquisition, preprocessing, and modeling.

