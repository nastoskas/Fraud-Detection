# Fraud Detection

Machine learning project for detecting fraudulent transactions. The dataset contains 30 features, and nine different classifiers were trained and compared, with a tuned **Random Forest** model achieving the best overall performance.

## Repository contents

| File | Description |
|---|---|
| `fraud_detection.ipynb` | Full workflow: exploratory data analysis, preprocessing, model training, comparison of classifiers, and hyperparameter tuning |
| `best_fraud_model_tuned.pkl` | Serialized, tuned Random Forest model (best performer) |
| `data_source.txt` | Link to the dataset used for training/evaluation |

## Approach

1. **Exploratory Data Analysis** — inspecting the 30 features and the class distribution
2. **Preprocessing** — scaling/cleaning the data and handling class imbalance
3. **Model comparison** — training and evaluating nine classifiers on the same splits
4. **Hyperparameter tuning** — optimizing the best-performing model (Random Forest)
5. **Evaluation** — comparing models using metrics suited to imbalanced classification (e.g. precision, recall, F1-score, ROC-AUC)

## Tech stack

- Python
- pandas / NumPy
- scikit-learn
- Jupyter Notebook

## Usage

```bash
git clone https://github.com/nastoskas/Fraud-Detection.git
cd Fraud-Detection
pip install -r requirements.txt   # or install pandas, scikit-learn, jupyter manually
jupyter notebook fraud_detection.ipynb
```

To use the saved model directly:

```python
import pickle

with open("best_fraud_model_tuned.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_new)
```

## Notes

- Dataset source is linked in `data_source.txt`.
- Feel free to adjust the classifier list, exact metrics, and dataset description above to match your final notebook results before pushing.
