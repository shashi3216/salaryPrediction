import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizations import Visualizer
from utils import generate_sample_data
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

def main():
    st.title("💰 Employee Salary Prediction System")
    st.markdown("### Machine Learning-powered salary prediction with interactive data analysis")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Upload & Analysis", "Model Training", "Salary Prediction", "Model Comparison"]
    )
    
    if page == "Data Upload & Analysis":
        data_upload_section()
    elif page == "Model Training":
        model_training_section()
    elif page == "Salary Prediction":
        prediction_section()
    elif page == "Model Comparison":
        comparison_section()

def data_upload_section():
    st.header("📊 Data Upload & Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your employee dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing employee data with salary information"
    )
    
    # Data generation options
    st.subheader("Pre-built Datasets")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sample Employee Dataset**")
        st.write("Generated dataset with realistic employee data including age, experience, department, education, and location factors.")
        if st.button("Generate Sample Dataset"):
            st.session_state.data = generate_sample_data()
            st.success("Sample dataset generated successfully!")
    
    with col2:
        st.write("**Adult Census Dataset**")
        st.write("Real-world census data with demographic and socioeconomic features converted to salary predictions.")
        if st.button("Load Adult Census Dataset"):
            try:
                st.session_state.data = pd.read_csv('adult_census_salary.csv')
                st.success("Adult Census dataset loaded successfully!")
            except FileNotFoundError:
                st.error("Adult Census dataset not found. Please generate it first.")
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Features", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            if 'salary' in data.columns:
                st.metric("Avg Salary", f"${data['salary'].mean():,.0f}")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Data analysis
        if st.checkbox("Show Data Analysis"):
            visualizer = Visualizer(data)
            
            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(data.describe(), use_container_width=True)
            
            # Salary distribution
            if 'salary' in data.columns:
                st.subheader("Salary Distribution")
                fig = visualizer.plot_salary_distribution()
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature correlations
                st.subheader("Feature Correlations")
                fig = visualizer.plot_correlation_matrix()
                st.plotly_chart(fig, use_container_width=True)
                
                # Salary by categorical features
                categorical_cols = data.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    st.subheader("Salary Analysis by Categories")
                    selected_feature = st.selectbox("Select feature for analysis:", categorical_cols)
                    fig = visualizer.plot_salary_by_category(selected_feature)
                    st.plotly_chart(fig, use_container_width=True)

def model_training_section():
    st.header("🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first in the 'Data Upload & Analysis' section.")
        return
    
    data = st.session_state.data
    
    # Check if salary column exists
    if 'salary' not in data.columns:
        st.error("Dataset must contain a 'salary' column for training.")
        return
    
    # Feature selection
    st.subheader("Feature Selection")
    available_features = [col for col in data.columns if col != 'salary']
    selected_features = st.multiselect(
        "Select features for training:",
        available_features,
        default=available_features
    )
    
    if not selected_features:
        st.warning("Please select at least one feature for training.")
        return
    
    # Model selection
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        algorithms = st.multiselect(
            "Select algorithms to train:",
            ["Linear Regression", "Random Forest", "XGBoost"],
            default=["Linear Regression", "Random Forest"]
        )
    
    with col2:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
    
    # Train models
    if st.button("Train Models", type="primary"):
        if not algorithms:
            st.warning("Please select at least one algorithm.")
            return
        
        with st.spinner("Training models..."):
            try:
                # Process data
                processor = DataProcessor()
                X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(
                    data, selected_features, test_size
                )
                
                # Train models
                trainer = ModelTrainer()
                results = {}
                
                for algorithm in algorithms:
                    model, metrics = trainer.train_model(
                        algorithm, X_train, X_test, y_train, y_test, cv_folds
                    )
                    results[algorithm] = {
                        'model': model,
                        'metrics': metrics,
                        'feature_names': feature_names
                    }
                
                st.session_state.models = results
                st.session_state.model_results = {k: v['metrics'] for k, v in results.items()}
                
                st.success("Models trained successfully!")
                
                # Display results
                st.subheader("Training Results")
                results_df = pd.DataFrame(st.session_state.model_results).T
                st.dataframe(results_df, use_container_width=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                for algorithm in algorithms:
                    if algorithm in ["Random Forest", "XGBoost"]:
                        model = results[algorithm]['model']
                        importance = model.feature_importances_
                        feature_imp_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importance
                        }).sort_values('importance', ascending=False)
                        
                        st.write(f"**{algorithm} Feature Importance:**")
                        fig = px.bar(
                            feature_imp_df, 
                            x='importance', 
                            y='feature',
                            orientation='h',
                            title=f"{algorithm} Feature Importance"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def prediction_section():
    st.header("🔮 Salary Prediction")
    
    if not st.session_state.models:
        st.warning("Please train models first in the 'Model Training' section.")
        return
    
    st.subheader("Enter Employee Information")
    
    # Get feature names from the first model
    feature_names = list(st.session_state.models.values())[0]['feature_names']
    
    # Create input form
    input_data = {}
    col1, col2 = st.columns(2)
    
    # Create dynamic input form based on actual features
    for i, feature in enumerate(feature_names):
        column = col1 if i % 2 == 0 else col2
        
        with column:
            # Handle different feature types dynamically
            feature_clean = feature.replace('_', ' ').replace('-', ' ').title()
            
            if 'age' in feature.lower():
                input_data[feature] = st.number_input(
                    f"{feature_clean}:", min_value=16, max_value=100, value=35
                )
            elif 'hour' in feature.lower():
                input_data[feature] = st.number_input(
                    f"{feature_clean}:", min_value=1, max_value=100, value=40
                )
            elif 'education' in feature.lower() and 'num' in feature.lower():
                input_data[feature] = st.number_input(
                    f"{feature_clean}:", min_value=1, max_value=16, value=9
                )
            elif 'capital' in feature.lower():
                input_data[feature] = st.number_input(
                    f"{feature_clean}:", min_value=0, value=0
                )
            elif 'fnlwgt' in feature.lower():
                input_data[feature] = st.number_input(
                    f"{feature_clean}:", min_value=10000, value=200000
                )
            elif feature in ['workclass']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov']
                )
            elif feature in ['education']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ['Bachelors', 'HS-grad', 'Some-college', 'Masters', 'Assoc-voc', 'Doctorate', 'Prof-school']
                )
            elif feature in ['marital_status', 'marital-status']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed']
                )
            elif feature in ['occupation']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Adm-clerical']
                )
            elif feature in ['relationship']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ['Husband', 'Wife', 'Own-child', 'Not-in-family', 'Other-relative', 'Unmarried']
                )
            elif feature in ['race']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
                )
            elif feature in ['gender']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:", ['Male', 'Female']
                )
            elif feature in ['native_country', 'native-country']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ['United-States', 'Canada', 'England', 'Germany', 'India', 'China', 'Mexico']
                )
            elif feature in ['experience_years']:
                input_data[feature] = st.number_input(
                    f"{feature_clean}:", min_value=0, max_value=50, value=5
                )
            elif feature in ['department', 'role']:
                input_data[feature] = st.text_input(
                    f"{feature_clean}:", value="Engineering"
                )
            elif feature in ['location']:
                input_data[feature] = st.selectbox(
                    f"{feature_clean}:",
                    ["New York", "San Francisco", "Los Angeles", "Chicago", "Boston"]
                )
            else:
                # Default to number input for unknown features
                input_data[feature] = st.number_input(
                    f"{feature_clean}:", value=0.0
                )
    
    # Model selection for prediction
    selected_model = st.selectbox(
        "Select model for prediction:",
        list(st.session_state.models.keys())
    )
    
    if st.button("Predict Salary", type="primary"):
        try:
            # Prepare input data
            processor = DataProcessor()
            input_df = pd.DataFrame([input_data])
            
            # Process input data (assuming we have the original data for encoding reference)
            processed_input = processor.process_single_prediction(input_df, st.session_state.data)
            
            # Make prediction
            model = st.session_state.models[selected_model]['model']
            prediction = model.predict(processed_input)[0]
            
            # Display prediction
            st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
            
            # Show prediction confidence (if available)
            metrics = st.session_state.models[selected_model]['metrics']
            st.info(f"Model R² Score: {metrics['r2_score']:.3f} | MAE: ${metrics['mae']:,.2f}")
            
            # Prediction comparison across all models
            if len(st.session_state.models) > 1:
                st.subheader("Prediction Comparison Across Models")
                comparison_data = []
                for model_name, model_info in st.session_state.models.items():
                    pred = model_info['model'].predict(processed_input)[0]
                    comparison_data.append({
                        'Model': model_name,
                        'Predicted Salary': f"${pred:,.2f}",
                        'R² Score': f"{model_info['metrics']['r2_score']:.3f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def comparison_section():
    st.header("📈 Model Comparison")
    
    if not st.session_state.model_results:
        st.warning("Please train models first in the 'Model Training' section.")
        return
    
    # Performance metrics comparison
    st.subheader("Performance Metrics Comparison")
    results_df = pd.DataFrame(st.session_state.model_results).T
    st.dataframe(results_df, use_container_width=True)
    
    # Visualize metrics
    metrics_to_plot = ['r2_score', 'mae', 'rmse']
    
    for metric in metrics_to_plot:
        if metric in results_df.columns:
            fig = px.bar(
                x=results_df.index,
                y=results_df[metric],
                title=f"{metric.upper().replace('_', ' ')} Comparison",
                labels={'x': 'Model', 'y': metric.upper()}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Best model recommendation
    st.subheader("Model Recommendation")
    best_model = results_df['r2_score'].idxmax()
    best_r2 = results_df.loc[best_model, 'r2_score']
    
    st.success(f"🏆 **Best Model:** {best_model}")
    st.info(f"**R² Score:** {best_r2:.3f}")
    
    # Cross-validation scores
    if 'cv_scores' in results_df.columns:
        st.subheader("Cross-Validation Scores")
        for model_name in results_df.index:
            cv_scores = st.session_state.model_results[model_name].get('cv_scores', [])
            if cv_scores:
                st.write(f"**{model_name}:** {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

if __name__ == "__main__":
    main()
