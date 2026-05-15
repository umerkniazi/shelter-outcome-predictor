# Shelter Outcome Predictor

A machine learning tool that predicts whether a shelter animal is likely to be adopted, transferred, or at risk of euthanasia — built on 62k+ real intake records from the Austin Animal Center.

## Overview

Trained a Random Forest classifier on animal attributes including age, sex status, species, and intake timing. The model uses class balancing to prioritize recall on at-risk animals, since missing a euthanasia case is costlier than a false alarm.

**Results (test set, 12.5k animals):**
- Overall accuracy: 76%
- Euthanasia recall: 63%
- Adoption F1: 0.83

## Features

- Age, sex status (neutered/intact/unknown), species, mixed breed, name presence
- Seasonal features: month and day of week
- Interactive Streamlit app for per-animal prediction with probability breakdown

## Stack

Python, scikit-learn, pandas, Streamlit

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

[Austin Animal Center Shelter Outcomes](https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-outcomes-and) via Kaggle.