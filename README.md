# Smart Product Pricing Challenge 💰

Machine Learning solution to predict e-commerce product prices using product descriptions (and optional images).

## Overview
This project predicts product prices using:
- Product title & description
- Quantity and numerical features
- Optional image features
- Ensemble Machine Learning models

**Goal:** Minimize SMAPE score for accurate price prediction.

---

## Features
✅ Feature Engineering  
✅ TF-IDF Text Processing  
✅ Ensemble Models (GBM + Random Forest + Ridge)  
✅ Optional XGBoost Advanced Model  
✅ Cross Validation  
✅ Optional Image Embeddings

---

## Project Structure

```bash
smart-product-pricing/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│
├── src/
│   └── utils.py
│
├── main_solution.py
├── advanced_solution.py
├── quick_eda.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/smart-product-pricing.git
cd smart-product-pricing
pip install -r requirements.txt
```

---

## Run Project

Basic Model:
```bash
python main_solution.py
```

Advanced Model:
```bash
python advanced_solution.py
```

EDA:
```bash
python quick_eda.py
```

Predictions will be saved in:

```bash
test_out.csv
```

---

## Models Used
- Gradient Boosting
- Random Forest
- Ridge Regression
- XGBoost (Advanced)

Ensemble helps improve accuracy and reduce overfitting.

---

## Results
Expected Performance:

- Basic Model SMAPE: ~12–18%
- Advanced Model SMAPE: ~10–15%

Much better than naive baselines.

---

## Key Features Extracted
- Pack Quantity (IPQ)
- Numeric attributes
- Brand keywords
- Size indicators
- TF-IDF text features
- Optional image embeddings

---

## Example Usage

```python
from main_solution import ProductPricePredictor
import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

model = ProductPricePredictor()
model.train(train)

predictions = model.predict(test)
```

---

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Transformers (optional)

---

## Evaluation Metric
Challenge metric:

**SMAPE (lower is better)**

---

## Future Improvements
- Better image feature extraction
- Hyperparameter tuning
- More advanced feature engineering

---

## License
MIT License

---

⭐ If you like this project, give it a star.



