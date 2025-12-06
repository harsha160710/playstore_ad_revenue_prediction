import pandas as pd
import sklearn
import os
from sklear.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklear.metrics import r2_score,mean_absolute_error,mean_squared_error

import xgboost as xgb
import joblib

from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError
from huggingface_hub import HfApi,,create_repo
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api=HfApi(token=os.getenv('HF_TOKEN_1'))

Xtrain_path = "hf://datasets/Harsha1001/play-store-ad-revenue-prediction/x_train.csv"
Xtest_path = "hf://datasets/Harsha1001/play-store-ad-revenue-prediction/x_test.csv"
ytrain_path = "hf://datasets/Harsha1001/play-store-ad-revenue-prediction/y_train.csv"
ytest_path = "hf://datasets/Harsha1001/play-store-ad-revenue-prediction/y_test.csv"

x_train = pd.read_csv(Xtrain_path)
x_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)

# Define numeric and categorical features
numeric_features = [
    'app_size_in_mb', 'price_in_usd', 'number_of_installs',
    'average_screen_time', 'active_users',
    'no_of_short_ads_per_hour', 'no_of_long_ads_per_hour'
]

categorical_features = [
    'app_category', 'free_or_paid', 'content_rating', 'screentime_category'
]

preprocessor= make_column_transformer(
    (StandardScaler(),numeric_features),
    (OneHotEncoder(handle_unknown='ignore'),categorical_features)
)

xgb_model=xgb.XGBRegressor(random_state=42)

param_grid={
    'xgbregressor__n_estimators':[50,75,100],
    'xgbregressor__max_depth':[3,4,5],
    'xgbregressor__learning_rate':[0.01,0.05],
    'xgbregressor__subsample':[0.5,0.6,0.7],
    'xgbregressor__colsample_bytree':[0.5,0.6,0.7],
    'xgbregressor__reg_lambda':[0.1,0.5,1]
}

model_pipeline=make_pipeline(preprocessor,xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(x_train)
    y_pred_test = best_model.predict(x_test)

    # Metrics
    train_rmse = mean_squared_error(y_train, y_pred_train)
    test_rmse = mean_squared_error(y_test, y_pred_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "train_R2": train_r2,
        "test_R2": test_r2
    })

    model_path="best_playstore_revenue_model_v1.joblib"
    joblib.dump(best_model,model_path)

    repo_id="Harsha1001/play-store-ad-revenue-prediction/playstore_revenue_model"
    repo_type="model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_playstore_revenue_model_v1.joblib",
        path_in_repo="best_playstore_revenue_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
