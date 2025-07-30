import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import seaborn as sns

class Visualizer:
    def __init__(self, data):
        self.data = data
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_salary_distribution(self):
        """
        Plot salary distribution histogram
        """
        fig = px.histogram(
            self.data,
            x='salary',
            title='Salary Distribution',
            labels={'salary': 'Salary ($)', 'count': 'Frequency'},
            marginal='box',
            nbins=30
        )
        
        fig.add_vline(
            x=self.data['salary'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${self.data['salary'].mean():,.0f}"
        )
        
        fig.add_vline(
            x=self.data['salary'].median(),
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: ${self.data['salary'].median():,.0f}"
        )
        
        fig.update_layout(
            showlegend=False,
            height=500
        )
        
        return fig
    
    def plot_correlation_matrix(self):
        """
        Plot correlation matrix for numerical features
        """
        # Select only numerical columns
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        fig.update_layout(
            height=600,
            width=600
        )
        
        return fig
    
    def plot_salary_by_category(self, category_column):
        """
        Plot salary distribution by categorical feature
        """
        if category_column not in self.data.columns:
            return None
        
        fig = px.box(
            self.data,
            x=category_column,
            y='salary',
            title=f'Salary Distribution by {category_column.replace("_", " ").title()}',
            points='outliers'
        )
        
        fig.update_layout(
            xaxis_title=category_column.replace("_", " ").title(),
            yaxis_title='Salary ($)',
            height=500
        )
        
        return fig
    
    def plot_feature_vs_salary(self, feature_column):
        """
        Plot feature vs salary scatter plot
        """
        if feature_column not in self.data.columns:
            return None
        
        if self.data[feature_column].dtype in ['object']:
            # Categorical feature
            fig = px.violin(
                self.data,
                x=feature_column,
                y='salary',
                title=f'Salary vs {feature_column.replace("_", " ").title()}'
            )
        else:
            # Numerical feature
            fig = px.scatter(
                self.data,
                x=feature_column,
                y='salary',
                title=f'Salary vs {feature_column.replace("_", " ").title()}',
                trendline='ols'
            )
        
        fig.update_layout(
            xaxis_title=feature_column.replace("_", " ").title(),
            yaxis_title='Salary ($)',
            height=500
        )
        
        return fig
    
    def plot_model_comparison(self, results_df):
        """
        Plot model performance comparison
        """
        metrics = ['r2_score', 'mae', 'rmse']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['R² Score', 'Mean Absolute Error', 'Root Mean Square Error']
        )
        
        for i, metric in enumerate(metrics, 1):
            if metric in results_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=results_df.index,
                        y=results_df[metric],
                        name=metric.upper(),
                        showlegend=False
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=400
        )
        
        return fig
    
    def plot_prediction_vs_actual(self, y_actual, y_predicted, model_name):
        """
        Plot predicted vs actual values
        """
        fig = px.scatter(
            x=y_actual,
            y=y_predicted,
            title=f'{model_name}: Predicted vs Actual Salaries',
            labels={'x': 'Actual Salary ($)', 'y': 'Predicted Salary ($)'}
        )
        
        # Add perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def plot_residuals(self, y_actual, y_predicted, model_name):
        """
        Plot residuals
        """
        residuals = y_actual - y_predicted
        
        fig = px.scatter(
            x=y_predicted,
            y=residuals,
            title=f'{model_name}: Residual Plot',
            labels={'x': 'Predicted Salary ($)', 'y': 'Residuals ($)'}
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(height=400)
        
        return fig
    
    def plot_feature_importance(self, importance_df, model_name):
        """
        Plot feature importance
        """
        # Take top 15 features for better visualization
        top_features = importance_df.head(15)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'{model_name}: Feature Importance',
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        
        fig.update_layout(
            height=max(400, len(top_features) * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_learning_curve(self, train_scores, val_scores, train_sizes):
        """
        Plot learning curve
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_scores,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_scores,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red')
            )
        )
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='R² Score',
            height=400
        )
        
        return fig
    
    def plot_cross_validation_scores(self, cv_results):
        """
        Plot cross-validation scores for different models
        """
        models = list(cv_results.keys())
        scores = [cv_results[model]['cv_scores'] for model in models]
        
        fig = go.Figure()
        
        for i, (model, score_list) in enumerate(zip(models, scores)):
            fig.add_trace(
                go.Box(
                    y=score_list,
                    name=model,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                )
            )
        
        fig.update_layout(
            title='Cross-Validation Scores Distribution',
            yaxis_title='R² Score',
            height=400
        )
        
        return fig
