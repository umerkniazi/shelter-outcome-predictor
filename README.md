# Shelter Outcome Predictor

A machine learning project that predicts whether a shelter animal is likely to be adopted, transferred or at risk of euthanasia using more than 62,000 historical intake records from the Austin Animal Center.

Rather than focusing solely on accuracy, the project explores an end-to-end machine learning workflow including data preparation, feature engineering, model comparison and evaluation. The final model prioritizes recall for at-risk animals, since missing a euthanasia case is more costly than a false alarm.

## Dataset

The model is trained on the **Austin Animal Center Shelter Outcomes** dataset, containing over 62,000 shelter intake records.

Features include:

- Species
- Age
- Sex status (neutered, intact or unknown)
- Mixed breed status
- Name presence
- Intake timing
- Seasonal features

## Methodology

Several classical machine learning models were evaluated, including:

- Logistic Regression
- K-Nearest Neighbors
- Decision Trees
- Random Forest

The final model uses a **Random Forest** classifier with class balancing to improve recall for higher-risk outcomes.

The project includes:

- Data preprocessing and feature engineering
- Model comparison
- Evaluation using multiple classification metrics
- A Streamlit interface for per-animal predictions with probability breakdowns

## Results

Evaluation was performed on a test set of approximately **12,500** animals.

| Metric | Value |
| :--- | ---: |
| Overall Accuracy | 76% |
| Euthanasia Recall | 63% |
| Adoption F1-score | 0.83 |

These results show that meaningful patterns can be learned from intake data, while also highlighting the limits of predicting outcomes influenced by many real-world factors.

## Limitations

The model relies only on information available at intake. Factors such as shelter resources, adoption campaigns, medical conditions and animal behavior are not represented.

Because the dataset comes from a single shelter, the learned patterns may not generalize well to other shelters or regions. Predictions should be viewed as decision-support rather than replacements for human judgment.

## Future Improvements

- Incorporate behavioral and medical information
- Evaluate models on data from multiple shelters
- Explore additional feature engineering techniques
- Improve model interpretability to better understand prediction factors

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

[Austin Animal Center Shelter Outcomes](https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-outcomes-and) via Kaggle.