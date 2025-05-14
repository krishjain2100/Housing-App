# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ===== Custom Transformers =====
from Custom_Transformers import CombinedAttributesAdder

# ===== Load and Prepare Data =====
housing = pd.read_csv(r"C:\Users\krish\OneDrive\Desktop\Machine Learning\datasets\housing\housing.csv")
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx]
    strat_test_set = housing.loc[test_idx]
housing = strat_train_set.drop(["median_house_value", "income_cat"], axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# ===== Pipelines =====
housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('scaler', StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# ===== Train and Save Model =====
param_grid = {
    "n_estimators": [10, 30],
    "max_features": [4, 6, 8]
}
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

final_model = grid_search.best_estimator_

# Save full pipeline (including preprocessing)
final_pipeline = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", final_model)
])
final_pipeline.fit(housing, housing_labels)

joblib.dump(final_pipeline, "housing_model.pkl")
print("✅ Model trained and saved to housing_model.pkl")
