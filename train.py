import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('data/aac_shelter_outcomes.csv')

    keep = ['Adoption', 'Transfer', 'Euthanasia']
    df = df[df['outcome_type'].isin(keep)].copy()

    df['has_name'] = df['name'].notna().astype(int)
    df = df.drop(columns=['animal_id', 'name', 'outcome_subtype', 'monthyear', 'date_of_birth'])

    def parse_age(age_str):
        if pd.isna(age_str):
            return None
        parts = age_str.split()
        value = int(parts[0])
        unit = parts[1]
        if 'year' in unit:
            return value * 365
        elif 'month' in unit:
            return value * 30
        elif 'week' in unit:
            return value * 7
        elif 'day' in unit:
            return value
        return None

    df['age_days'] = df['age_upon_outcome'].apply(parse_age)
    df['is_mix'] = df['breed'].str.contains('Mix', case=False).astype(int)
    df['is_neutered'] = df['sex_upon_outcome'].str.contains('Neutered|Spayed', case=False).astype(int)
    df['is_intact'] = df['sex_upon_outcome'].str.contains('Intact', case=False).astype(int)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek

    df = df.dropna(subset=['age_days'])

    features = ['age_days', 'is_mix', 'is_neutered', 'is_intact', 'has_name', 'month', 'day_of_week']
    X = pd.get_dummies(df[features + ['animal_type']], columns=['animal_type'])
    y = df['outcome_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/shelter_model.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
    print("Done.")

if __name__ == '__main__':
    train()