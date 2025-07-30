import pandas as pd
import numpy as np

def create_adult_salary_dataset():
    """
    Create Adult Census dataset with estimated salary values
    """
    # Column names for the Adult dataset
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    # Sample data based on the Adult Census dataset structure
    # This creates a representative dataset with real patterns
    np.random.seed(42)
    
    data = []
    n_samples = 5000
    
    # Define realistic categories
    workclasses = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
    educations = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
    marital_statuses = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    occupations = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    relationships = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    races = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    genders = ['Female', 'Male']
    countries = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    
    for i in range(n_samples):
        # Generate basic demographics
        age = np.random.randint(17, 91)
        workclass = np.random.choice(workclasses, p=[0.7, 0.08, 0.05, 0.03, 0.06, 0.04, 0.02, 0.02])
        education = np.random.choice(educations)
        marital_status = np.random.choice(marital_statuses)
        occupation = np.random.choice(occupations)
        relationship = np.random.choice(relationships)
        race = np.random.choice(races, p=[0.85, 0.03, 0.03, 0.03, 0.06])
        gender = np.random.choice(genders)
        native_country = np.random.choice(countries, p=[0.9] + [0.1/40]*40)
        
        # Generate work-related features
        hours_per_week = np.random.normal(40, 12)
        hours_per_week = max(1, min(99, int(hours_per_week)))
        
        capital_gain = np.random.exponential(100) if np.random.random() < 0.1 else 0
        capital_loss = np.random.exponential(50) if np.random.random() < 0.05 else 0
        
        fnlwgt = np.random.randint(10000, 1500000)
        
        # Educational number mapping
        education_num_map = {
            'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
            '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
            'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
            'Prof-school': 15, 'Doctorate': 16
        }
        educational_num = education_num_map.get(education, 9)
        
        # Calculate salary based on realistic factors
        base_salary = 25000
        
        # Age factor
        if age < 25:
            age_multiplier = 0.8
        elif age < 35:
            age_multiplier = 1.0
        elif age < 45:
            age_multiplier = 1.3
        elif age < 55:
            age_multiplier = 1.5
        else:
            age_multiplier = 1.2
        
        # Education factor
        education_multipliers = {
            'Preschool': 0.5, '1st-4th': 0.6, '5th-6th': 0.7, '7th-8th': 0.8, '9th': 0.85,
            '10th': 0.9, '11th': 0.95, '12th': 1.0, 'HS-grad': 1.1, 'Some-college': 1.3,
            'Assoc-voc': 1.4, 'Assoc-acdm': 1.5, 'Bachelors': 1.8, 'Masters': 2.2,
            'Prof-school': 2.8, 'Doctorate': 3.0
        }
        education_multiplier = education_multipliers.get(education, 1.0)
        
        # Work hours factor
        hours_multiplier = min(2.0, hours_per_week / 40)
        
        # Occupation factor
        occupation_multipliers = {
            'Exec-managerial': 2.0, 'Prof-specialty': 1.8, 'Tech-support': 1.5,
            'Sales': 1.3, 'Craft-repair': 1.2, 'Adm-clerical': 1.0,
            'Machine-op-inspct': 1.1, 'Transport-moving': 1.1, 'Protective-serv': 1.3,
            'Other-service': 0.8, 'Farming-fishing': 0.9, 'Handlers-cleaners': 0.7,
            'Priv-house-serv': 0.6, 'Armed-Forces': 1.2
        }
        occupation_multiplier = occupation_multipliers.get(occupation, 1.0)
        
        # Gender factor (reflecting historical wage gaps in the dataset period)
        gender_multiplier = 1.0 if gender == 'Male' else 0.85
        
        # Calculate final salary
        estimated_salary = (base_salary * age_multiplier * education_multiplier * 
                          hours_multiplier * occupation_multiplier * gender_multiplier)
        
        # Add capital gains/losses
        estimated_salary += capital_gain - capital_loss
        
        # Add some randomness
        noise = np.random.normal(0, estimated_salary * 0.1)
        final_salary = max(15000, estimated_salary + noise)
        
        data.append({
            'age': age,
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education': education,
            'educational_num': educational_num,
            'marital_status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'gender': gender,
            'capital_gain': int(capital_gain),
            'capital_loss': int(capital_loss),
            'hours_per_week': hours_per_week,
            'native_country': native_country,
            'salary': round(final_salary, 2)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create the dataset
    df = create_adult_salary_dataset()
    
    # Save to CSV
    df.to_csv('adult_census_salary.csv', index=False)
    print(f"Dataset created with {len(df)} records")
    print("\nDataset info:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    print(f"\nSalary statistics:")
    print(df['salary'].describe())