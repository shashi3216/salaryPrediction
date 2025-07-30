import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_names = []
    
    def prepare_data(self, data, feature_columns, test_size=0.2, random_state=42):
        """
        Prepare data for machine learning training
        """
        try:
            # Create feature matrix and target vector
            X = data[feature_columns].copy()
            y = data['salary'].copy()
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Identify categorical and numerical columns
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
            numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_columns),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
                ]
            )
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Fit and transform the data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Store the preprocessor for later use
            self.column_transformer = preprocessor
            
            # Get feature names
            feature_names = self._get_feature_names(preprocessor, numerical_columns, categorical_columns)
            self.feature_names = feature_names
            
            return X_train_processed, X_test_processed, y_train, y_test, feature_names
            
        except Exception as e:
            raise Exception(f"Error in data preparation: {str(e)}")
    
    def process_single_prediction(self, input_data, original_data):
        """
        Process single input for prediction
        """
        try:
            if self.column_transformer is None:
                raise Exception("Data processor not fitted. Please train models first.")
            
            # Handle missing values
            input_processed = self._handle_missing_values(input_data)
            
            # Transform using the fitted preprocessor
            processed_input = self.column_transformer.transform(input_processed)
            
            return processed_input
            
        except Exception as e:
            raise Exception(f"Error processing input data: {str(e)}")
    
    def _handle_missing_values(self, data):
        """
        Handle missing values in the dataset
        """
        data_copy = data.copy()
        
        # Fill numerical columns with median
        numerical_columns = data_copy.select_dtypes(exclude=['object']).columns
        for col in numerical_columns:
            if data_copy[col].isnull().any():
                data_copy[col].fillna(data_copy[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_columns = data_copy.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data_copy[col].isnull().any():
                mode_value = data_copy[col].mode()
                if len(mode_value) > 0:
                    data_copy[col].fillna(mode_value[0], inplace=True)
                else:
                    data_copy[col].fillna('Unknown', inplace=True)
        
        return data_copy
    
    def _get_feature_names(self, preprocessor, numerical_columns, categorical_columns):
        """
        Get feature names after preprocessing
        """
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(numerical_columns)
        
        # Add categorical feature names (one-hot encoded)
        try:
            # Get the one-hot encoder
            ohe = preprocessor.named_transformers_['cat']
            if hasattr(ohe, 'get_feature_names_out'):
                cat_features = ohe.get_feature_names_out(categorical_columns)
            else:
                # Fallback for older versions
                cat_features = [f"{col}_{val}" for col in categorical_columns 
                              for val in ohe.categories_[categorical_columns.get_loc(col)][1:]]
            feature_names.extend(cat_features)
        except:
            # If we can't get the exact names, use generic ones
            feature_names.extend([f"cat_feature_{i}" for i in range(len(categorical_columns))])
        
        return feature_names
    
    def get_data_info(self, data):
        """
        Get basic information about the dataset
        """
        info = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numerical_columns': data.select_dtypes(exclude=['object']).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist()
        }
        return info
    
    def validate_data(self, data):
        """
        Validate if the data is suitable for machine learning
        """
        issues = []
        
        # Check if salary column exists
        if 'salary' not in data.columns:
            issues.append("Dataset must contain a 'salary' column")
        
        # Check for sufficient data
        if len(data) < 10:
            issues.append("Dataset too small (minimum 10 rows required)")
        
        # Check for features
        feature_columns = [col for col in data.columns if col != 'salary']
        if len(feature_columns) == 0:
            issues.append("No feature columns found")
        
        # Check for variance in target variable
        if 'salary' in data.columns:
            if data['salary'].nunique() < 2:
                issues.append("Salary column has insufficient variance")
        
        return issues
