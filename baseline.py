#!/usr/bin/env python
# coding: utf-8

# # Pipeline for Machine Learning Model

import argparse
import openml
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import load_iris, fetch_california_housing
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,FunctionTransformer,LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score,mean_squared_error,make_scorer

warnings.filterwarnings('ignore')


IdentityEncoding = FunctionTransformer(lambda x: x)
hidden_size = ()

# Generate values for the hidden sizes of the NN and append to tuple
for i in range(200, 0, -50):
    hidden_size += (i,)

encoders = {'minmax':MinMaxScaler(),'scaler':StandardScaler(),  
                'identity':IdentityEncoding
           }



# List of results to create a dataframe
result_list = []
def run(X_train, y_train, X_test, y_test, model, encoder, task='classification'):
    numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('encoder', encoders[encoder])
    ])

    categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])
    if task == 'classification':
        classifier = models[model]
    elif task == 'regression':
        regressor = models[model]

    # Define column transformer to apply transformers to appropriate columns
    preprocessor = ColumnTransformer([
    ('num', numerical_transformer, X_train.select_dtypes(include=['number']).columns),
    ('cat', categorical_transformer, X_train.select_dtypes(include=['object']).columns)
    ])
    if task == 'classification':
        pipeline = Pipeline([('preprocessor', preprocessor),
                         ('classifier', classifier)])
    else:
        pipeline = Pipeline([('preprocessor', preprocessor),
                         ('regressor', regressor)])
    # Fit pipeline to training data
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = np.array(y_pred_train)
    y_pred_test = np.array(y_pred_test)

    # Evaluate on test data
    if task == 'classification':
        accuracy_train = accuracy_score(y_train,y_pred_train)
        #roc_train = roc_auc_score(y_train, y_pred_train)
        precision_train =precision_score(y_train, y_pred_train, average='weighted')
        precision_train =precision_score(y_train, y_pred_train, average='weighted')
        accuracy_test = accuracy_score(y_test, y_pred_test) 
        #roc_test = roc_auc_score(y_test, y_pred_test)
        precision_test =precision_score(y_test, y_pred_test, average='weighted')
        precision_test =precision_score(y_test, y_pred_test, average='weighted')
        new_row = {'Model': model,
        'Encoder': encoder,
        'Test Accuracy': accuracy_test,
        #'Test ROC AUC': roc_test,
        'Test Precision':precision_test}
    elif task == 'regression':
        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
        new_row =  {'Model': model,
        'Encoder': encoder,
        'Test RMSE': rmse_test}

    # Append new row to result list
    
    result_list.append(new_row)
    
    # Print scores
    if task == 'classification': 
        print('\n')
        print(f'Model: {model}\nEncoder: {encoder}')
        print('Train Accuracy', accuracy_train)
        #print('Train ROC AUC Score', roc_train)
        print('Train Precision Score', precision_train)
        print('Test Accuracy:', accuracy_test)
        #print('Test ROC AUC Score', roc_test)
        print('Test Precision Score', precision_test)
    else:
        print('\n')
        print(f'Model: {model}\nEncoder: {encoder}')
        print('Train RMSE', rmse_train)
        print('Test RMSE', rmse_test)
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply encodings and train a model on a dataset.')
    parser.add_argument('--dataset', type=str, help='Input dataset file')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'], default='classification',
                        help='Task type (regression or classification)')

    args = parser.parse_args()
    # Get dataset from OpenML
    if args.dataset == 'house_16h':
        dataset_id = 574
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = pd.DataFrame(data=X, columns=dataset.features)
        df['target'] = y
    elif args.dataset == 'adult':
        dataset_id = 1590
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = X
        df['target'] = y
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].astype('object')
    elif args.dataset == 'gesture':
        dataset_id = 4538
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = X
        df['target'] = y
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].astype('object')
    elif args.dataset == 'microsoft':
        dataset_id = 45579
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = X
        df['target'] = y
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].astype('object')
    elif args.dataset == 'covtype':
        dataset_id = 150
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = X
        df['target'] = y
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].astype('object')
    elif args.dataset == 'otto':
        dataset_id = 45548
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = X
        df['target'] = y
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].astype('object')
    elif args.dataset == 'santander':
        dataset_id = 45566
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = X
        df['target'] = y
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].astype('object')
    elif args.dataset == 'higgs':
        dataset_id = 23512
        # Download the dataset by its ID
        # Retrieve the dataset features and labels
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Print some information about the dataset
        print(f"Dataset name: {dataset.name}")
        print(f"Number of instances: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        df = X
        df['target'] = y
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].astype('object')
    elif args.dataset == 'california':
        #Load the California Housing dataset
        data = fetch_california_housing()
        # Create a DataFrame from the dataset
        df = pd.DataFrame(data.data, columns=data.feature_names)
    # Add the target variable to the DataFrame
        df['target'] = data.target
        print(f"Dataset name: California Housing")
        print(f"Number of instances: {df.values.shape[0]}")
        print(f"Number of features: {df.values.shape[1] - 1}")
    task = args.task
    if task == 'regression':
        models = {'ridge':Ridge(), \
                  'svm':SVR(), \
                  'nn':MLPRegressor(hidden_layer_sizes=hidden_size, activation='relu', solver='adam', max_iter=20), \
                 }
    elif task == 'classification':
        models = {'ridge':RidgeClassifier(), \
                  'svm':SVC(), \
                  'nn':MLPClassifier(hidden_layer_sizes=hidden_size, activation='relu', solver='adam', max_iter=20), \
                 }
    if df['target'].dtype == 'object':
        df['target'] = LabelEncoder().fit_transform(df['target'])
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42)
    for i in encoders:
        for j in models:
            run(X_train, y_train, X_test, y_test,model=j, encoder=i, task=task)
    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(result_list)
    # Save the results to the specified output file
    if args.task == 'regression':
        new_df = results_df
        new_df = new_df.sort_values(by=['Test RMSE'], ascending=True)
    else:
        new_df = results_df
        new_df = new_df.sort_values(by=['Test Accuracy'], ascending=False)
    new_df.to_csv(f"Results/baseline_results_{args.dataset}.csv", index=False)

    print(f"Scores saved to Results folder as baseline_results_{args.dataset}.csv")

