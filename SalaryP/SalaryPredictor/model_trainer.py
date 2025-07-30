import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            )
        }
    
    def train_model(self, algorithm, X_train, X_test, y_train, y_test, cv_folds=5):
        """
        Train a specific machine learning model
        """
        try:
            # Get the model
            model = self.models[algorithm]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            metrics['cv_scores'] = cv_scores.tolist()
            
            return model, metrics
            
        except Exception as e:
            raise Exception(f"Error training {algorithm}: {str(e)}")
    
    def _calculate_metrics(self, y_train, y_test, y_pred_train, y_pred_test):
        """
        Calculate various performance metrics
        """
        metrics = {}
        
        # Training metrics
        metrics['train_r2'] = r2_score(y_train, y_pred_train)
        metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        # Testing metrics
        metrics['r2_score'] = r2_score(y_test, y_pred_test)
        metrics['mae'] = mean_absolute_error(y_test, y_pred_test)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Additional metrics
        metrics['mape'] = self._mean_absolute_percentage_error(y_test, y_pred_test)
        
        return metrics
    
    def _mean_absolute_percentage_error(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error
        """
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def get_model_info(self, algorithm):
        """
        Get information about a specific model
        """
        model = self.models.get(algorithm)
        if model is None:
            return None
        
        info = {
            'name': algorithm,
            'type': type(model).__name__,
            'parameters': model.get_params()
        }
        
        return info
    
    def compare_models(self, results):
        """
        Compare multiple trained models
        """
        comparison = {}
        
        for model_name, model_data in results.items():
            metrics = model_data['metrics']
            comparison[model_name] = {
                'R² Score': metrics['r2_score'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape'],
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std']
            }
        
        return pd.DataFrame(comparison).T
    
    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def predict_with_confidence(self, model, X, confidence_level=0.95):
        """
        Make predictions with confidence intervals (for supported models)
        """
        predictions = model.predict(X)
        
        # For Random Forest, we can calculate prediction intervals
        if hasattr(model, 'estimators_'):
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)
            
            return predictions, lower_bound, upper_bound
        
        return predictions, None, None
    
    def validate_model_performance(self, metrics, min_r2=0.5, max_mae_ratio=0.2):
        """
        Validate if model performance meets minimum criteria
        """
        issues = []
        
        # Check R² score
        if metrics['r2_score'] < min_r2:
            issues.append(f"Low R² score: {metrics['r2_score']:.3f} (minimum: {min_r2})")
        
        # Check if model is overfitting
        train_r2 = metrics.get('train_r2', 0)
        test_r2 = metrics['r2_score']
        if train_r2 - test_r2 > 0.2:
            issues.append(f"Possible overfitting: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")
        
        # Check cross-validation consistency
        cv_std = metrics.get('cv_std', 0)
        if cv_std > 0.1:
            issues.append(f"High cross-validation variance: {cv_std:.3f}")
        
        return issues
