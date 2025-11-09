import logging
import os
from datetime import datetime

import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
path = os.environ.get('PROJECT_PATH', '.')

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'id', 'url', 'region', 'region_url', 'price',
        'manufacturer', 'image_url', 'description',
        'posting_date', 'lat', 'long'
    ]
    return df.drop(columns_to_drop, axis=1)

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        return q25 - 1.5 * iqr, q75 + 1.5 * iqr

    df = df.copy()
    low, high = calculate_outliers(df['year'])
    df.loc[df['year'] < low, 'year'] = round(low)
    df.loc[df['year'] > high, 'year'] = round(high)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    def short_model(x):
        return x.lower().split(' ')[0] if not pd.isna(x) else x

    df = df.copy()
    df['short_model'] = df['model'].apply(short_model)
    df['age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df

def pipeline() -> None:
    df = pd.read_csv(f'{path}/data/train/homework.csv')
    X = df.drop('price_category', axis=1)
    y = df['price_category']

    num_features = make_column_selector(dtype_include=['int64', 'float64'])
    cat_features = make_column_selector(dtype_include=object)

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = Pipeline([
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('columns', ColumnTransformer([
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]))
    ])

    models = [
        LogisticRegression(solver='lbfgs', max_iter=1000),
        RandomForestClassifier(),
        SVC()
    ]

    best_score, best_pipe = 0.0, None
    for model in models:
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        logging.info(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score, best_pipe = score.mean(), pipe

    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    best_pipe.fit(X, y)

    model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'
    with open(model_filename, 'wb') as f:
        dill.dump(best_pipe, f)
    logging.info(f'Model is saved as {model_filename}')

if __name__ == '__main__':
    print("Старт...")
    pipeline()
    print(" Выполнено!")
