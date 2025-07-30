import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=1000):
    """
    Generate sample employee dataset for demonstration
    """
    np.random.seed(42)
    random.seed(42)
    
    # Define categories
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'Product']
    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    locations = ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston', 'Seattle', 'Austin']
    roles = {
        'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
        'Sales': ['Sales Rep', 'Senior Sales Rep', 'Sales Manager', 'Sales Director'],
        'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Marketing Director', 'CMO'],
        'HR': ['HR Specialist', 'HR Manager', 'HR Director', 'CHRO'],
        'Finance': ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'CFO'],
        'Operations': ['Operations Specialist', 'Operations Manager', 'Operations Director', 'COO'],
        'Product': ['Product Manager', 'Senior PM', 'Product Director', 'CPO']
    }
    
    data = []
    
    for i in range(n_samples):
        # Generate basic info
        age = np.random.normal(35, 8)
        age = max(22, min(65, int(age)))
        
        experience_years = max(0, min(age - 22, int(np.random.normal(8, 5))))
        
        department = np.random.choice(departments)
        role = np.random.choice(roles[department])
        education = np.random.choice(education_levels)
        location = np.random.choice(locations)
        
        # Generate salary based on various factors
        base_salary = 50000
        
        # Experience factor
        experience_bonus = experience_years * 2000
        
        # Education factor
        education_multipliers = {
            "High School": 1.0,
            "Bachelor's": 1.2,
            "Master's": 1.4,
            "PhD": 1.6
        }
        education_bonus = base_salary * (education_multipliers[education] - 1)
        
        # Department factor
        department_multipliers = {
            'Engineering': 1.4,
            'Product': 1.3,
            'Sales': 1.2,
            'Finance': 1.15,
            'Marketing': 1.1,
            'Operations': 1.05,
            'HR': 1.0
        }
        department_bonus = base_salary * (department_multipliers[department] - 1)
        
        # Location factor
        location_multipliers = {
            'San Francisco': 1.5,
            'New York': 1.4,
            'Seattle': 1.3,
            'Boston': 1.2,
            'Los Angeles': 1.15,
            'Austin': 1.1,
            'Chicago': 1.05
        }
        location_bonus = base_salary * (location_multipliers[location] - 1)
        
        # Role seniority factor
        role_multipliers = {
            'Software Engineer': 1.0, 'Senior Engineer': 1.3, 'Tech Lead': 1.6, 'Engineering Manager': 2.0,
            'Sales Rep': 1.0, 'Senior Sales Rep': 1.2, 'Sales Manager': 1.5, 'Sales Director': 2.2,
            'Marketing Specialist': 1.0, 'Marketing Manager': 1.4, 'Marketing Director': 1.8, 'CMO': 3.0,
            'HR Specialist': 1.0, 'HR Manager': 1.3, 'HR Director': 1.7, 'CHRO': 2.5,
            'Financial Analyst': 1.0, 'Senior Analyst': 1.2, 'Finance Manager': 1.5, 'CFO': 2.8,
            'Operations Specialist': 1.0, 'Operations Manager': 1.3, 'Operations Director': 1.6, 'COO': 2.7,
            'Product Manager': 1.2, 'Senior PM': 1.5, 'Product Director': 1.9, 'CPO': 2.9
        }
        role_multiplier = role_multipliers.get(role, 1.0)
        
        # Calculate final salary
        calculated_salary = (base_salary + experience_bonus + education_bonus + 
                           department_bonus + location_bonus) * role_multiplier
        
        # Add some randomness
        noise = np.random.normal(0, calculated_salary * 0.1)
        final_salary = max(30000, calculated_salary + noise)
        
        data.append({
            'age': age,
            'experience_years': experience_years,
            'department': department,
            'role': role,
            'education_level': education,
            'location': location,
            'salary': round(final_salary, 2)
        })
    
    return pd.DataFrame(data)

def validate_uploaded_data(data):
    """
    Validate uploaded CSV data
    """
    issues = []
    
    # Check if it's a DataFrame
    if not isinstance(data, pd.DataFrame):
        issues.append("Invalid data format")
        return issues
    
    # Check minimum rows
    if len(data) < 10:
        issues.append("Dataset should have at least 10 rows")
    
    # Check for salary column
    if 'salary' not in data.columns:
        issues.append("Dataset must contain a 'salary' column")
    
    # Check salary column data type
    if 'salary' in data.columns:
        try:
            pd.to_numeric(data['salary'])
        except:
            issues.append("Salary column must contain numeric values")
    
    # Check for at least one feature column
    feature_columns = [col for col in data.columns if col != 'salary']
    if len(feature_columns) == 0:
        issues.append("Dataset must contain at least one feature column")
    
    # Check for excessive missing values
    missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    if missing_percentage > 50:
        issues.append(f"Dataset has too many missing values ({missing_percentage:.1f}%)")
    
    return issues

def format_currency(value):
    """
    Format numeric value as currency
    """
    return f"${value:,.2f}"

def calculate_salary_statistics(salary_series):
    """
    Calculate comprehensive salary statistics
    """
    stats = {
        'count': len(salary_series),
        'mean': salary_series.mean(),
        'median': salary_series.median(),
        'std': salary_series.std(),
        'min': salary_series.min(),
        'max': salary_series.max(),
        'q25': salary_series.quantile(0.25),
        'q75': salary_series.quantile(0.75),
        'iqr': salary_series.quantile(0.75) - salary_series.quantile(0.25)
    }
    return stats

def detect_outliers(salary_series, method='iqr'):
    """
    Detect salary outliers using IQR method
    """
    if method == 'iqr':
        Q1 = salary_series.quantile(0.25)
        Q3 = salary_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = salary_series[(salary_series < lower_bound) | (salary_series > upper_bound)]
        return outliers.index.tolist()
    
    return []

def suggest_feature_engineering(data):
    """
    Suggest feature engineering opportunities
    """
    suggestions = []
    
    # Check for date columns that could be converted to age/tenure
    for col in data.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            suggestions.append(f"Consider converting {col} to tenure/age features")
    
    # Check for categorical columns with high cardinality
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].nunique() > 20:
            suggestions.append(f"Consider grouping categories in {col} (high cardinality: {data[col].nunique()})")
    
    # Check for potential interaction features
    numerical_cols = data.select_dtypes(exclude=['object']).columns
    if len(numerical_cols) >= 2:
        suggestions.append("Consider creating interaction features between numerical variables")
    
    return suggestions

def export_model_report(model_results, data_info):
    """
    Generate a comprehensive model report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_summary': {
            'total_records': data_info['shape'][0],
            'features': data_info['shape'][1] - 1,  # excluding target
            'missing_values': sum(data_info['missing_values'].values())
        },
        'model_performance': model_results,
        'best_model': max(model_results.keys(), key=lambda k: model_results[k]['r2_score'])
    }
    return report

def clean_column_names(data):
    """
    Clean column names for better processing
    """
    data_copy = data.copy()
    
    # Convert to lowercase and replace spaces with underscores
    data_copy.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in data_copy.columns]
    
    # Remove special characters
    data_copy.columns = [col.replace('(', '').replace(')', '').replace('.', '_') for col in data_copy.columns]
    
    return data_copy

def get_sample_data_info():
    """
    Get information about the sample dataset structure
    """
    info = {
        'description': 'Sample employee dataset with salary information',
        'features': [
            'age: Employee age (22-65)',
            'experience_years: Years of work experience',
            'department: Department (Engineering, Sales, Marketing, etc.)',
            'role: Job role within department',
            'education_level: Education level (High School to PhD)',
            'location: Work location (major US cities)',
            'salary: Annual salary in USD (target variable)'
        ],
        'size': '1000 records',
        'use_case': 'Demonstration and testing of ML models'
    }
    return info
