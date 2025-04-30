###############################################################################
# SECTION 1: LIBRARY IMPORTS AND INITIAL SETUP
# This section imports all the necessary libraries for data analysis, machine learning,
# data visualization, and web application functionality
###############################################################################

# Importing streamlit - the main library for creating web applications
import streamlit as st
# Importing pandas - library for data manipulation and analysis
import pandas as pd
# Importing numpy - library for numerical computations
import numpy as np
# Importing matplotlib - library for creating visualizations
import matplotlib.pyplot as plt
# Importing ensemble models from scikit-learn - for machine learning algorithms
from sklearn import ensemble
# Importing train_test_split - to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Importing r2_score - a metric to evaluate model performance
from sklearn.metrics import r2_score
# Importing seaborn - a library for statistical data visualization
import seaborn as sns

# Importing additional models from scikit-learn
# LinearRegression, Ridge, Lasso, ElasticNet - different regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# RandomForestRegressor, AdaBoostRegressor - ensemble methods for regression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
# SVR - Support Vector Regression model
from sklearn.svm import SVR
# KNeighborsRegressor - K-Nearest Neighbors regression model
from sklearn.neighbors import KNeighborsRegressor
# MLPRegressor - Multi-layer Perceptron (neural network) for regression
from sklearn.neural_network import MLPRegressor
# Metrics for evaluating model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error
# RandomizedSearchCV - for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV    

# Importing preprocessing modules for handling categorical features
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importing Adaptive Best Subset Selection model
from abess.linear import LinearRegression as AbessLinearRegression
# For checking if a module is available
import importlib.util
# These are commented out, were likely used for a QLattice model
# import feyn
# from feyn.qlattice import QLattice
                        
# Importing OpenAI for AI-powered insights
import openai
# For environment variables
import os

###############################################################################
# SECTION 2: AUTHENTICATION SYSTEM
# This section handles user authentication to access the application
###############################################################################

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False  # Default: user is not authenticated

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Check if the username exists in the credentials store
        if st.session_state["username"] in st.secrets.get("credentials", {}):
            # Check if the password matches
            if st.session_state["password"] == st.secrets["credentials"][st.session_state["username"]]:
                # Set authenticated to True if credentials are correct
                st.session_state.authenticated = True
                # Store current user in session state
                st.session_state["current_user"] = st.session_state["username"]
                # Remove password from session state for security
                del st.session_state["password"]
                return True
            else:
                # Password is incorrect
                st.session_state.authenticated = False
                st.error("😕 Password incorrect")
                return False
        else:
            # Username doesn't exist
            st.session_state.authenticated = False
            st.error("😕 User not found")
            return False

    # First run or if user logs out
    if not st.session_state.authenticated:
        # Show login page title
        st.title("Advertising Data Science Assistant")
        st.markdown("Please enter your username and password")
        
        # Create login form with username and password fields
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submit = st.form_submit_button("Login")
            
            # Check credentials when form is submitted
            if submit:
                password_entered()

    # Return current authentication status
    return st.session_state.authenticated

###############################################################################
# SECTION 3: MAIN APPLICATION
# This section contains the main application functionality that runs after
# successful authentication. It handles file uploads, data processing, and 
# visualization.
###############################################################################

# Check authentication before showing the main app
if check_password():
    # Set up Streamlit main title
    st.title("Data Science Assistant")
    
    # Show logged in user in the sidebar
    st.sidebar.success(f"Logged in as: {st.session_state.get('current_user', 'User')}")
    
    # Add a logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()  # Restart the app
    
    # Check for OpenAI API key in secrets
    if 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
        # Use API key from secrets if available
        api_key = st.secrets['openai']['api_key']
        st.success("✅ OpenAI API key configured")
    else:
        # Otherwise ask the user to enter it manually
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if not api_key:
            st.warning("Please enter an OpenAI API key to use this application.")
    
    # Store API key in environment variable if provided
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        
        # LLM model selection dropdown
        model = st.selectbox(
            "Select the model",
            ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"]
        )
        
        # Session management - allows users to maintain separate data sessions
        session_id = st.text_input("Session ID", value="default_session")
        
        # Initialize session state for storage
        if 'store' not in st.session_state:
            st.session_state.store = {}

        # Create tabs for different functionalities
        tabs = st.tabs(["Data Analysis", "ROI Prediction"])
        
        # Tab 1: Data Analysis
        with tabs[0]:
            # File uploader for CSV files
            uploaded_file = st.file_uploader("Choose your .csv file", type="csv")
            
            # DataFrame and column handling - only runs if a file is uploaded
            if uploaded_file is not None:
                try:
                    # Read the CSV file into a pandas DataFrame
                    df = pd.read_csv(uploaded_file)
                    
                    # Store dataframe in session state for access in other tabs
                    st.session_state['df'] = df
                    
                    # Display the column names from the CSV
                    st.write("Columns in the uploaded CSV file:")
                    columns = df.columns.tolist()
                    for col in columns:
                        st.write(col)
                     
                    # Feature selection UI components
                    ID = st.selectbox("Select the ID variable", columns)
                    target = st.selectbox("Select the target variable", columns)
                    # Create a list of features excluding target and ID columns
                    features_ls = [col for col in columns if col not in [target, ID]]
                    
                    # Store important variables in session state
                    st.session_state['ID'] = ID
                    st.session_state['target'] = target
                    st.session_state['features_ls'] = features_ls
                    
                    # Let user select which features to use
                    features = st.multiselect("Select the features", features_ls)
                    
                    # Store selected features in session state
                    st.session_state['features'] = features
                    
                    # Identify categorical features automatically
                    if features:
                        # Identify categorical features (object or category dtype)
                        categorical_features = [f for f in features if df[f].dtype == 'object' or df[f].dtype.name == 'category']
                        # Store categorical features in session state
                        st.session_state['categorical_features'] = categorical_features
                        
                        # Display identified categorical features
                        if categorical_features:
                            st.write("Detected categorical features:")
                            for col in categorical_features:
                                st.write(f"- {col} (unique values: {len(df[col].unique())})")
                        else:
                            st.write("No categorical features detected")
                    
                    # Additional options that only appear if features are selected
                    if features:
                        # Select variable to convert to quantiles (for visualization)
                        var_convert_range = st.selectbox("Select the variable to convert to quantiles", features)
                        # Select variable to group by (for visualization)
                        var_groupby = st.selectbox("Select the variable to group by", features)
                        # Number of quantiles to create
                        convert_range = st.slider("How many quantiles to convert to", min_value=2, max_value=20, value=10)
                        
                        ###############################################################################
                        # SECTION 4: DATA VISUALIZATION AND ANALYSIS FUNCTION
                        # This large function handles data visualization, machine learning model training,
                        # and prediction generation for the application
                        ###############################################################################
                        
                        # Visualization function definition
                        def generate_visualizations(data, 
                                                test_size=0.3,       # test size
                                                random_state=13,     # random state for reproducibility
                                                var_convert_range=var_convert_range,  # variable to convert to quantiles
                                                var_groupby=var_groupby,  # variable to group by
                                                convert_range=convert_range,  # how many quantiles to convert to
                                                outcome_var=target,  # outcome variable
                                                id_var=ID,          # ID variable
                                                # Machine Learning Model Parameters:
                                                n_estimators=400,
                                                max_depth=4,
                                                min_samples_split=10,
                                                learning_rate=0.01,
                                                loss='squared_error',
                                                features=features,
                                                categorical_features=st.session_state['categorical_features']):  # Explicitly pass categorical_features
                            """
                            Generate visualizations for the data
                            """
                            
                            # Create a variable to store positive donors
                            positive_donors = []
                            
                            # Split data into train and test sets with stratification
                            # If a categorical feature is selected as groupby, use it for stratification
                            if var_groupby in categorical_features and var_groupby in data.columns:
                                st.write(f"Using stratified split on {var_groupby}")
                                train, test = train_test_split(data, test_size=test_size, 
                                                             random_state=random_state,
                                                             stratify=data[var_groupby])
                            else:
                                # Regular split without stratification
                                train, test = train_test_split(data, test_size=test_size, 
                                                             random_state=random_state)
                            
                            # --------- Visualization 1: Donation by Income ---------
                            plt.figure(figsize=(12,6))  # Set figure size
                            np.random.seed(123)  # Set random seed for reproducibility
                            
                            try:
                                # Check if var_convert_range is numerical for qcut (quantile cut)
                                if pd.api.types.is_numeric_dtype(train[var_convert_range]):
                                    # Create quantiles of the variable
                                    train_with_quantiles = train.assign(
                                        income_quantile=pd.qcut(train[var_convert_range], q=convert_range)
                                    )
                                    # Create bar plot of target variable by quantile
                                    sns.barplot(data=train_with_quantiles,
                                            x="income_quantile", y=outcome_var)
                                    plt.title(f"{outcome_var} by {var_convert_range}")
                                    plt.xticks(rotation=70)  # Rotate x labels for readability
                                    image1 = st.pyplot(plt)  # Display plot in Streamlit
                                else:
                                    st.warning(f"{var_convert_range} is not numerical. Cannot create quantiles.")
                                    image1 = None
                            except Exception as e:
                                st.error(f"Error in Visualization 1: {e}")
                                image1 = None
                            
                            # --------- Visualization 2: Donation by Location (Historical Data) ---------
                            try:
                                plt.figure(figsize=(12,6))  # Set figure size
                                np.random.seed(123)  # Set random seed for reproducibility
                                
                                # Sum donations by location
                                location_donation = train.groupby(var_groupby)[outcome_var].sum()
                                # Set bar colors based on positive/negative values
                                colors = ['blue' if value >= 0 else 'red' for value in location_donation]
                                
                                # Create bar plot of donations summed by location
                                sns.barplot(data=train, x=var_groupby, y=outcome_var, estimator=sum, ci=None, palette=colors)
                                plt.title(f"{outcome_var} by {var_groupby} - Historical (Train) Data")
                                plt.xticks(rotation=45)  # Rotate x labels for readability
                                image2 = st.pyplot(plt)  # Display plot in Streamlit
                            except Exception as e:
                                st.error(f"Error in Visualization 2: {e}")
                                image2 = None
                            
                            # --------- Identify Locations for Investment ---------
                            try:
                                # Calculate mean target value by location
                                location_to_net = train.groupby(var_groupby)[outcome_var].mean().to_dict()
                                # Filter to only locations with positive mean values
                                location_to_invest = {location: net for location, net in location_to_net.items() if net > 0}
                                st.write(f"{var_groupby} to invest in based on historical data: {list(location_to_invest.keys())}")
                                # Handle case where no locations meet criteria
                                if not location_to_invest:
                                    st.warning("No locations meet investment criteria.")
                                    return image1, image2, None, None, None, []
                                    
                                # Filter test data to only rows with locations that meet investment criteria
                                location_policy = test[test[var_groupby].isin(location_to_invest.keys())]
                                
                                # Handle empty dataframe case
                                if location_policy.empty:
                                    st.warning("No test data matches the location investment criteria.")
                                    return image1, image2, None, None, None, []
                                
                                # --------- Visualization 3: Location Policy on Test Data ---------
                                plt.figure(figsize=(12,6))  # Set figure size
                                np.random.seed(123)  # Set random seed for reproducibility
                                
                                # Sum target values by location in filtered test data
                                location_donation = location_policy.groupby(var_groupby)[outcome_var].sum()
                                # Set bar colors based on positive/negative values
                                colors = ['green' if value >= 0 else 'red' for value in location_donation]
                                
                                # Create bar plot of target values summed by location
                                sns.barplot(data=location_policy, x=var_groupby, y=outcome_var, estimator=sum, ci=None, palette=colors)
                                plt.title(f"{var_groupby} Policy when Applied to Test (Unseen) Data")
                                plt.xticks(rotation=45)  # Rotate x labels for readability
                                image3 = st.pyplot(plt)  # Display plot in Streamlit
                                # Show locations with positive target values
                                st.write(f"Potential positive value {var_groupby}s: {location_donation[location_donation > 0].index.tolist()}")
                                
                                # --------- Visualization 4: Histogram of Donations ---------
                                plt.figure(figsize=(10,6))  # Set figure size
                                # Create histogram of target values
                                sns.histplot(data=location_policy, x=outcome_var)
                                # Show average target value in title
                                plt.title(f"Average {outcome_var}: {location_policy[outcome_var].sum() / test.shape[0]:.2f}")
                                image4 = st.pyplot(plt)  # Display plot in Streamlit
                            except Exception as e:
                                st.error(f"Error in Visualizations 3-4: {e}")
                                image3, image4 = None, None
                                return image1, image2, None, None, None, []
                            
                            # --------- Machine Learning Model for Policy Prediction ---------
                            try:
                                # Identify categorical features
                                categorical_features = [f for f in features if f in st.session_state['categorical_features']]
                                numerical_features = [f for f in features if f not in categorical_features]
                                
                                # Define function to preprocess data with categorical feature handling
                                def preprocess_data(df, is_training=True):
                                    # Display debugging info
                                    st.write("Preprocessing data, is_training =", is_training)
                                    
                                    # Create a copy of the dataframe with selected features
                                    X = df[features].copy()
                                    
                                    # Store categorical encodings during training for reuse in prediction
                                    if is_training:
                                        # Initialize dictionary for all possible dummy columns
                                        all_dummy_columns = {}
                                        
                                    # Handle categorical features using one-hot encoding (if any)
                                    if categorical_features:
                                        for col in categorical_features:
                                            if col in X.columns:
                                                if is_training:
                                                    # For training: Get all possible values from the entire dataset (not just train)
                                                    # This ensures we capture all possible categories
                                                    all_values = data[col].dropna().unique()
                                                    
                                                    # Manually create dummy column names for all possible values
                                                    # This avoids the drop_first issue and ensures consistency
                                                    dummy_columns = [f"{col}_{val}" for val in all_values]
                                                    
                                                    # Store these dummy column names for future prediction
                                                    all_dummy_columns[col] = dummy_columns
                                                    
                                                    # Store in session state
                                                    st.session_state['all_dummy_columns'] = all_dummy_columns
                                                    
                                                    # Create dummies and make sure all expected columns exist
                                                    dummies = pd.get_dummies(X[col], prefix=col)
                                                    
                                                    # Debug
                                                    st.write(f"Training: Created dummy columns for {col}:", dummies.columns.tolist())
                                                else:
                                                    # For prediction: Use the same column names as in training
                                                    if 'all_dummy_columns' in st.session_state and col in st.session_state['all_dummy_columns']:
                                                        expected_dummy_columns = st.session_state['all_dummy_columns'][col]
                                                        
                                                        # Create dummies
                                                        dummies = pd.get_dummies(X[col], prefix=col)
                                                        
                                                        # Debug
                                                        st.write(f"Prediction: Created dummy columns for {col}:", dummies.columns.tolist())
                                                        st.write(f"Expected columns:", expected_dummy_columns)
                                                        
                                                        # Explicitly add missing dummy columns
                                                        for dummy_col in expected_dummy_columns:
                                                            if dummy_col not in dummies.columns:
                                                                st.write(f"Adding missing column: {dummy_col}")
                                                                dummies[dummy_col] = 0
                                                    else:
                                                        # If no encoding info available, just create dummies (should not happen)
                                                        dummies = pd.get_dummies(X[col], prefix=col)
                                                        st.warning(f"No encoding information available for {col}")
                                                
                                                # Add dummy columns to dataframe
                                                X = pd.concat([X, dummies], axis=1)
                                                # Drop original categorical column
                                                X = X.drop(col, axis=1)
                                                
                                    # Map location to its mean target value (if var_groupby is in features)
                                    if var_groupby in X.columns and var_groupby in location_to_net:
                                        X[var_groupby] = X[var_groupby].map(location_to_net)
                                    
                                    # Handle missing values by replacing with mean
                                    X = X.fillna(X.mean())
                                    
                                    # If training, store the feature columns for future prediction
                                    if is_training:
                                        # Store column names in the order they appear
                                        st.session_state['processed_feature_columns'] = X.columns.tolist()
                                        st.write("Stored feature columns:", st.session_state['processed_feature_columns'])
                                    else:
                                        # If predicting, ensure all columns from training exist
                                        if 'processed_feature_columns' in st.session_state:
                                            expected_cols = st.session_state['processed_feature_columns']
                                            
                                            # Find missing columns
                                            missing_cols = [col for col in expected_cols if col not in X.columns]
                                            if missing_cols:
                                                st.write("Missing columns detected:", missing_cols)
                                                for col in missing_cols:
                                                    X[col] = 0
                                            
                                            # Handle extra columns
                                            extra_cols = [col for col in X.columns if col not in expected_cols]
                                            if extra_cols:
                                                st.write("Extra columns detected:", extra_cols)
                                                X = X.drop(columns=extra_cols, errors='ignore')
                                            
                                            # Force column order to match training
                                            X = X[expected_cols]
                                            
                                            # Verify columns match
                                            if list(X.columns) == expected_cols:
                                                st.success("Column check passed!")
                                            else:
                                                st.error("Column mismatch after processing!")
                                                st.write("Expected:", expected_cols)
                                                st.write("Got:", list(X.columns))
                                    
                                    return X
                                
                                # Set target variable name
                                tar = outcome_var
                                
                                # Preprocess training data
                                X_train_processed = preprocess_data(train, is_training=True)
                                
                                # Create a validation set for model selection (80% train, 20% validation)
                                X_train, X_val, y_train, y_val = train_test_split(
                                    X_train_processed, train[tar], test_size=0.2, random_state=42
                                )
                                
                                # Store column names for future prediction
                                st.session_state['processed_feature_columns'] = X_train_processed.columns.tolist()
                                
                                # Store the location_to_net mapping for future prediction
                                st.session_state['location_to_net'] = location_to_net
                                
                                # Define hyperparameter search spaces for each model
                                param_distributions = {
                                    "Linear Regression": {},  # No hyperparameters to tune
                                    
                                    "Ridge Regression": {
                                        'alpha': np.logspace(-3, 3, 20),  # Regularization strength values to try
                                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']  # Solver algorithms to try
                                    },
                                    
                                    "Lasso Regression": {
                                        'alpha': np.logspace(-3, 3, 20),  # Regularization strength values to try
                                        'selection': ['cyclic', 'random']  # Feature selection methods to try
                                    },
                                    
                                    "ElasticNet": {
                                        'alpha': np.logspace(-3, 3, 20),  # Regularization strength values to try
                                        'l1_ratio': np.linspace(0.1, 0.9, 9),  # L1 ratio values to try (mix of L1 and L2 penalties)
                                        'selection': ['cyclic', 'random']  # Feature selection methods to try
                                    },
                                    
                                    "Random Forest": {
                                        'n_estimators': [50, 100, 200, 300, 400],  # Number of trees to try
                                        'max_depth': [None, 5, 10, 15, 20, 25],  # Maximum tree depth values to try
                                        'min_samples_split': [2, 5, 10, 15],  # Min samples required to split a node
                                        'min_samples_leaf': [1, 2, 4, 8],  # Min samples required at a leaf node
                                        'max_features': ['auto', 'sqrt', 'log2', None]  # Number of features to consider at splits
                                    },
                                    
                                    "AdaBoost": {
                                        'n_estimators': [50, 100, 200, 300],  # Number of estimators to try
                                        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],  # Learning rate values to try
                                        'loss': ['linear', 'square', 'exponential']  # Loss function types to try
                                    },
                                    
                                    # "SVR": {  # Support Vector Regression - commented out
                                    #     'C': np.logspace(-3, 2, 6),
                                    #     'gamma': np.logspace(-3, 2, 6),
                                    #     'kernel': ['linear', 'poly', 'rbf'],
                                    #     'epsilon': [0.01, 0.1, 0.2]
                                    # },
                                    
                                    "KNN": {
                                        'n_neighbors': list(range(1, 31)),  # Number of neighbors to try
                                        'weights': ['uniform', 'distance'],  # Weight functions to try
                                        'p': [1, 2]  # Distance metric (1=Manhattan, 2=Euclidean)
                                    },
                                    
                                    # "Neural Network": {  # Multi-layer Perceptron - commented out
                                    #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                                    #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                    #     'solver': ['lbfgs', 'sgd', 'adam'],
                                    #     'alpha': np.logspace(-5, 3, 5),
                                    #     'learning_rate': ['constant', 'invscaling', 'adaptive']
                                    # },
                                    
                                    "Gradient Boosting": {
                                        'n_estimators': [100, 200, 300, 400, 500],  # Number of boosting stages to try
                                        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],  # Learning rate values to try
                                        'max_depth': [3, 4, 5, 6, 7, 8],  # Maximum tree depth values to try
                                        'min_samples_split': [2, 5, 10, 15, 20],  # Min samples required to split
                                        'min_samples_leaf': [1, 2, 4, 8],  # Min samples required at a leaf node
                                        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Fraction of samples to use for trees
                                        'loss': ['squared_error', 'absolute_error', 'huber']  # Loss function types to try
                                    }
                                }
                                
                                # Define base models to evaluate
                                base_models = {
                                    "Linear Regression": LinearRegression(),
                                    "Ridge Regression": Ridge(),
                                    "Lasso Regression": Lasso(),
                                    "ElasticNet": ElasticNet(),
                                    "Random Forest": RandomForestRegressor(random_state=42),
                                    "AdaBoost": AdaBoostRegressor(random_state=42),
                                    #"SVR": SVR(),  # Commented out
                                    "KNN": KNeighborsRegressor(),
                                    #"Neural Network": MLPRegressor(random_state=42, max_iter=1000),  # Commented out
                                    "Gradient Boosting": ensemble.GradientBoostingRegressor(random_state=42)
                                }
                                
                                # Skip Abess model due to dimension problems
                                try:
                                    # Comment out Abess model to avoid X.shape[1] errors
                                    # base_models["Abess (Best Subset)"] = AbessLinearRegression(fit_intercept=True)
                                    # param_distributions["Abess (Best Subset)"] = {
                                    #     'support_size': list(range(1, min(12, X_train.shape[1]) + 1)),
                                    #     'ic_type': ['aic', 'bic', 'ebic', 'gic']
                                    # }
                                    # Log that we're skipping this model
                                    st.info("Skipping Abess model to avoid dimension compatibility issues.")
                                except ImportError:
                                    st.warning("Abess package is not installed. To include Abess in model evaluation, install it using: pip install abess")
                                    
                                # Create empty list to store model evaluation results
                                results = []
                                
                                # Number of iterations for RandomizedSearchCV
                                n_iter = 20
                                
                                # Set cross-validation folds
                                cv = 3
                                
                                # Total models to evaluate
                                total_models = len(base_models)
                                
                                # Initialize a container to hold hyperparameter results
                                hyperparams_container = st.container()
                                
                                # Define a session state for hyperparams if it doesn't exist
                                if 'hyperparams' not in st.session_state:
                                    st.session_state.hyperparams = {}
                                
                                # Create a progress bar to show model evaluation progress
                                progress_bar = st.progress(0)
                                st.write("Evaluating different machine learning models with hyperparameter optimization...")
                                
                                # Evaluate each model with hyperparameter tuning
                                for i, (name, model) in enumerate(base_models.items()):
                                    try:
                                        # Update progress bar
                                        progress_bar.progress((i) / total_models)
                                        
                                        # Store that we're optimizing this model
                                        if name not in st.session_state.hyperparams:
                                            st.session_state.hyperparams[name] = {"optimizing": True}
                                        
                                        # Get parameter space for this model
                                        param_space = param_distributions.get(name, {})
                                        
                                        # If empty parameter space or QLattice, skip optimization
                                        if not param_space or name == "QLattice":
                                            # Just fit and evaluate the model as is
                                            model.fit(X_train, y_train)
                                            best_model = model
                                            
                                            # Store this information
                                            st.session_state.hyperparams[name]["no_params"] = True
                                            st.session_state.hyperparams[name]["message"] = f"Model {name} has no hyperparameters to tune or uses internal optimization."
                                        else:
                                            # Use RandomizedSearchCV for hyperparameter optimization
                                            # Calculate actual number of iterations based on parameter space size
                                            n_actual_iter = min(n_iter, np.prod([len(v) if hasattr(v, '__len__') else 1 
                                                                                for v in param_space.values()]))
                                            
                                            # If SVR with non-linear kernel, add additional gamma options
                                            if name == "SVR":
                                                param_grid_svr = param_space.copy()
                                                if 'gamma' in param_grid_svr and 'kernel' in param_grid_svr:
                                                    param_grid_svr = {k: (v if k != 'gamma' else ['scale', 'auto'] + list(v)) 
                                                                    for k, v in param_grid_svr.items()}
                                                else:
                                                    param_grid_svr = param_space
                                                
                                                # Create search object for SVR
                                                search = RandomizedSearchCV(model, param_grid_svr, 
                                                                        n_iter=n_actual_iter, 
                                                                        cv=cv, 
                                                                        scoring='r2', 
                                                                        random_state=42,
                                                                        n_jobs=-1)  # Use all available CPU cores
                                            else:
                                                # Create search object for other models
                                                search = RandomizedSearchCV(model, param_space, 
                                                                        n_iter=n_actual_iter, 
                                                                        cv=cv, 
                                                                        scoring='r2', 
                                                                        random_state=42,
                                                                        n_jobs=-1)  # Use all available CPU cores
                                            
                                            # Fit the search object to find best parameters
                                            search.fit(X_train, y_train)
                                            best_model = search.best_estimator_
                                            
                                            # Store the best parameters
                                            st.session_state.hyperparams[name]["has_params"] = True
                                            st.session_state.hyperparams[name]["best_params"] = search.best_params_
                                        
                                        # Evaluate best model on training and validation data
                                        y_train_pred = best_model.predict(X_train)
                                        y_val_pred = best_model.predict(X_val)
                                        
                                        # Calculate performance metrics
                                        train_r2 = r2_score(y_train, y_train_pred)  # R² on training data
                                        val_r2 = r2_score(y_val, y_val_pred)  # R² on validation data
                                        mse = mean_squared_error(y_val, y_val_pred)  # Mean squared error
                                        mae = mean_absolute_error(y_val, y_val_pred)  # Mean absolute error
                                        
                                        # Add model results to results list
                                        results.append({
                                            'Model': name,
                                            'Train R²': train_r2,
                                            'Validation R²': val_r2,
                                            'MSE': mse,
                                            'MAE': mae,
                                            'model_object': best_model  # Save the trained model object
                                        })
                                        
                                        st.write(f"✅ Successfully evaluated {name}")
                                        
                                    except Exception as e:
                                        st.warning(f"❌ Error evaluating {name}: {str(e)}")
                                        # Continue with other models instead of stopping
                                
                                # Clear progress bar when finished
                                progress_bar.empty()
                                
                                # Create DataFrame of results for display (exclude model objects)
                                results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model_object'} for r in results])
                                # Sort by validation R² in descending order
                                results_df = results_df.sort_values('Validation R²', ascending=False).reset_index(drop=True)
                                
                                # Get the best model based on validation R²
                                if not results_df.empty:
                                    best_model_idx = results_df['Validation R²'].idxmax()
                                    best_model_name = results_df.loc[best_model_idx, 'Model']
                                    best_model = [r['model_object'] for r in results if r['Model'] == best_model_name][0]
                                    
                                    # Show success message with best model name and performance
                                    st.success(f"Best (Optimized) ML model's performance on data unseen to it: R² = {results_df.loc[best_model_idx, 'Validation R²']:.4f}")
                                    
                                    # Store the best model in session state for later use in prediction tab
                                    st.session_state['best_model'] = best_model
                                    st.session_state['best_model_name'] = best_model_name

                                    # Use the best model for final predictions
                                    reg = best_model
                                    # Fit the model on all training data
                                    reg.fit(X_train_processed, train[tar])
                                else:
                                    # If no models were successfully trained, use default Gradient Boosting
                                    st.error("No models were successfully trained. Using default Gradient Boosting model.")
                                    reg = ensemble.GradientBoostingRegressor(
                                        n_estimators=100, 
                                        max_depth=3, 
                                        learning_rate=0.1, 
                                        random_state=42
                                    )
                                    # Fit the default model
                                    reg.fit(X_train_processed, train[tar])
                                    
                                    # Store the default model in session state
                                    st.session_state['best_model'] = reg
                                    st.session_state['best_model_name'] = "Default Gradient Boosting"
                                
                                # Make predictions on training data
                                train_pred = X_train_processed.copy()
                                train_pred['predictions'] = reg.predict(X_train_processed)
                                
                                # Preprocess test data
                                X_test_processed = preprocess_data(test)
                                
                                # Make predictions on test data
                                ml_policy = test.assign(prediction=reg.predict(X_test_processed))

                                # Extract records with positive predictions
                                positive_donors_df = ml_policy[ml_policy["prediction"] > 0].copy()
                                
                                # Use the ID variable passed as a parameter
                                # Check if ID column exists, otherwise use index
                                if id_var in positive_donors_df.columns:
                                    current_id_column = id_var
                                else:
                                    # Create an ID using the index if no ID column exists
                                    positive_donors_df['generated_id'] = positive_donors_df.index
                                    current_id_column = 'generated_id'  # Assumes that the index is unique
                                
                                # Calculate the difference between prediction and actual
                                positive_donors_df['diff'] = (positive_donors_df['prediction'] - positive_donors_df[outcome_var])
                                # Calculate percentage difference
                                positive_donors_df['diff_percent'] = abs(positive_donors_df['diff'] / positive_donors_df[outcome_var].replace(0, 0.001)) * 100
                                
                                # Filter to records where prediction is close to actual value
                                # Close is defined as less than 20% difference, or prediction < 0.1 if actual value is 0
                                close_prediction_donors = positive_donors_df[
                                    ((positive_donors_df[outcome_var] != 0) & (positive_donors_df['diff_percent'] < 20)) | 
                                    ((positive_donors_df[outcome_var] == 0) & (positive_donors_df['prediction'] < 0.1))
                                ].copy()
                                
                                # Create a list of dictionaries with donor info
                                positive_donors = close_prediction_donors[[current_id_column
                                                                        #, outcome_var
                                                                        , 'prediction'
                                                                        #, 'diff'
                                                                        , 'diff_percent'
                                                                        ]].to_dict('records')
                                
                                # Add a rank based on closeness of prediction (smaller difference is better)
                                positive_donors_sorted = sorted(positive_donors, key=lambda x: x['diff_percent'])
                                for rank, donor in enumerate(positive_donors_sorted, 1):
                                    donor['rank'] = rank
                                
                                # Calculate R² scores for train and test sets
                                train_r2 = r2_score(y_true=train[tar], y_pred=train_pred["predictions"])
                                test_r2 = r2_score(y_true=test[tar], y_pred=reg.predict(X_test_processed))
                                
                                # Display R² scores
                                st.write(f"Train R²: {train_r2:.4f}")
                                st.write(f"Test R²: {test_r2:.4f}")
                                
                                # --------- Visualization 5: ML vs. Location-Based Policy ---------
                                plt.figure(figsize=(10,6))  # Set figure size
                                # Filter test data to records with positive ML predictions
                                ml_df = ml_policy[ml_policy["prediction"] > 0]
                                # Filter test data to records with positive locations
                                locations_df = ml_policy[ml_policy[var_groupby].isin(location_to_invest.keys())]
                                
                                # Handle empty dataframes
                                if ml_df.empty:
                                    st.warning("ML policy produced no positive predictions.")
                                    return image1, image2, image3, image4, None, []
                                    
                                # Create histograms comparing ML policy vs Location policy
                                sns.histplot(data=ml_df, x=outcome_var, color="C1", label="ML Policy")
                                sns.histplot(data=locations_df, x=outcome_var, label=f"{var_groupby} Policy")
                                
                                # Title showing average target values for each policy
                                plt.title(f"ML/AI {outcome_var}: {ml_df[outcome_var].sum() / test.shape[0]:.2f} | "
                                        f"{var_groupby} Policy {outcome_var}: {locations_df[outcome_var].sum() / test.shape[0]:.2f}")
                                
                                plt.legend()  # Show legend
                                image5 = st.pyplot(plt)  # Display plot in Streamlit
                                
                                # Return all visualizations and positive donors
                                return image1, image2, image3, image4, image5, positive_donors
                            except Exception as e:
                                st.error(f"Error in Machine Learning Model: {e}")
                                return image1, image2, image3, image4, None, []
                        
                        ###############################################################################
                        # SECTION 5: VISUALIZATION GENERATION AND RESULTS DISPLAY
                        # This section handles visualization generation and displaying results to users
                        ###############################################################################
                        
                        # Button to generate visualizations
                        if st.button("Generate Visualizations"):
                            with st.spinner("Generating...."):
                                # Call the visualization function with selected parameters
                                i1, i2, i3, i4, i5, positive_donors = generate_visualizations(
                                    df,
                                    test_size=0.3,
                                    random_state=13,
                                    var_convert_range=var_convert_range,
                                    var_groupby=var_groupby,
                                    convert_range=convert_range,
                                    outcome_var=target,
                                    n_estimators=400,
                                    max_depth=4,
                                    min_samples_split=10,
                                    learning_rate=0.01,
                                    loss='squared_error',
                                    features=features
                                )

                                # Store positive donors in session state for later use
                                if positive_donors:
                                    st.session_state.positive_donors = positive_donors
                                    st.success(f"Found {len(positive_donors)} {ID}s with difference between prediction and actual {target} < 20%")
                                    
                                    # Show top 10 donors in a table
                                    st.subheader("Top 10 campaign with Most Accurate Predictions")
                                    top_donors_df = pd.DataFrame(positive_donors[:10])
                                    # Format percentage columns for better readability
                                    if 'diff_percent' in top_donors_df.columns:
                                        top_donors_df['diff_percent'] = top_donors_df['diff_percent'].map('{:.2f}%'.format)
                                    st.dataframe(top_donors_df)
                                    
                                    # Add download button for full list of donors
                                    full_donors_df = pd.DataFrame(positive_donors)
                                    if 'diff_percent' in full_donors_df.columns:
                                        full_donors_df['diff_percent'] = full_donors_df['diff_percent'].map(lambda x: '{:.2f}'.format(x))
                                    # Convert dataframe to CSV format
                                    csv = full_donors_df.to_csv(index=False).encode('utf-8')
                                    # Create download button
                                    st.download_button(
                                        f"Download this table as CSV (Please see disclaimer below)",
                                        csv,
                                        f"{ID}_list_with_predicted_{target}.csv",
                                        "text/csv",
                                        key='download-csv'
                                    )
                                # This would save visualizations if that function exists (commented out)
                                # if 'save_visualizations' in locals() or 'save_visualizations' in globals():
                                #     save_visualizations(i1, i2, i3, i4, i5)
                        
                        ###############################################################################
                        # SECTION 6: CHATBOT INTERFACE FOR DATA ANALYSIS
                        # This section provides an AI-powered chat interface for asking questions
                        # about the data and getting insights
                        ###############################################################################
                        
                        # GPT Analysis section
                        st.header("Chat Bot")
                        st.info("Ask questions about the data and get AI-powered insights")
                        # Create container for bot responses
                        response_area = st.container()
                        
                        # Function to generate analysis based on user queries
                        def generate_analysis(user_query, data):
                            try:
                                # Initialize OpenAI client with API key
                                client = openai.OpenAI(api_key=api_key)
                                
                                # Prepare dataset summary for context
                                data_summary = f"""
                                Dataset shape: {data.shape}
                                Columns: {', '.join(data.columns.tolist())}
                                Target variable: {target}
                                Selected features: {', '.join(features)}
                                
                                Data sample:
                                {data.head(5).to_string()}
                                
                                Statistical summary:
                                {data[features + [target]].describe().to_string()}
                                """
                                
                                # Add categorical feature information if available
                                if 'categorical_features' in st.session_state and st.session_state['categorical_features']:
                                    cat_features = st.session_state['categorical_features']
                                    cat_info = "Categorical features:\n"
                                    for feat in cat_features:
                                        if feat in data.columns:
                                            unique_values = data[feat].nunique()
                                            sample_values = data[feat].unique()[:5].tolist()
                                            cat_info += f"- {feat}: {unique_values} unique values, examples: {sample_values}\n"
                                    data_summary += f"\n\n{cat_info}"

                                # Add positive donors information if available
                                positive_donors_info = ""
                                if 'positive_donors' in st.session_state and st.session_state.positive_donors:
                                    donors = st.session_state.positive_donors
                                    try:
                                        # Basic donor summary information
                                        positive_donors_info = f"""
                                        \n\nAccurate prediction donors summary:
                                        - Total donors with accurate predictions: {len(donors)}
                                        - Average predicted donation: {sum(d['prediction'] for d in donors) / len(donors):.2f}
                                        """
                                        
                                        # Only add fields if they exist in all donors
                                        if all('diff_percent' in d for d in donors):
                                            positive_donors_info += f"- Average difference percentage: {sum(float(d['diff_percent']) for d in donors) / len(donors):.2f}%\n"
                                        
                                        # Add information about top 5 donors
                                        positive_donors_info += "\nTop 5 donors with relatively more accurate predictions:"
                                        
                                        for i, donor in enumerate(donors[:5], 1):
                                            # Find the ID field more safely (in case column name changes)
                                            id_field = next((key for key in donor.keys() if 'id' in key.lower()), "unknown_id")
                                            
                                            # Build info string with only guaranteed fields
                                            donor_info = f"\n{i}. ID {donor[id_field]}: Predicted {donor['prediction']:.2f}"
                                            
                                            # Only add diff_percent if it exists
                                            if 'diff_percent' in donor:
                                                donor_info += f", Difference: {float(donor['diff_percent']):.2f}%"
                                            
                                            positive_donors_info += donor_info
                                            
                                    except Exception as e:
                                        st.warning(f"Error processing donor information: {e}")
                                        # Provide simplified donor info if there was an error
                                        positive_donors_info = f"\n\nFound {len(donors)} donors with positive predictions."
                                
                                # Add ML model information if available
                                ml_info = ""
                                if 'best_model_name' in st.session_state:
                                    ml_info = f"\n\nBest performing model: {st.session_state['best_model_name']}"
                                    
                                    # Add hyperparameter info if available
                                    if 'hyperparams' in st.session_state and st.session_state['best_model_name'] in st.session_state['hyperparams']:
                                        model_params = st.session_state['hyperparams'][st.session_state['best_model_name']]
                                        if 'best_params' in model_params:
                                            ml_info += f"\nOptimized parameters: {model_params['best_params']}"
                                
                                # Add ML rationale context if keyword is detected in the query
                                ml_rationale = ""
                                if any(keyword in user_query.lower() for keyword in ["why", "better", "rationale", "explain", "compared", "advantage", "ml policy", "machine learning"]):
                                    # Detailed explanation for why ML might be better than simple location-based policy
                                    ml_rationale = f"""
                                    \n\nRATIONALE FOR ML PERFORMANCE:
                                    
                                    1. {var_groupby} policy uses {ID} location as the only deciding factor. The policy doesn't consider other important details about individual {id}s.
                                    
                                    2. ML-based policy considers {ID} behavior, previous {target} history, {features} and {var_groupby}n all together - identifies complex patterns and interactions between these features.
                                    
                                    3. ML builds a more nuanced understanding of what makes someone a "high net {ID}."
                                    
                                    4. The ML algorithm used here can detect subtle, nonlinear relationships between {features}.
                                    
                                    5. Empirical evidence: In testing, ML Policy achieved a net {target} average higher than {var_groupby} Policy.
                                    """
                                            
                                # Send the user query to the LLM with context
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are a data scientist who analyzes data. Provide insights, patterns, and recommendations based on the data."},
                                        {"role": "user", "content": f"Dataset context:\n{data_summary}{positive_donors_info}{ml_info}{ml_rationale}\n\nUser question: {user_query}"}
                                    ]
                                )
                                # Return the LLM's response
                                return response.choices[0].message.content
                            except Exception as e:
                                return f"Error generating analysis: {str(e)}"
                        
                        # User input for analysis
                        user_query = st.text_input("Ask a question about your data:")
                        
                        # Generate and display response when user submits a query
                        if user_query:
                            with st.spinner("Analyzing data..."):
                                # Call the analysis function with the user's query
                                analysis = generate_analysis(user_query, df)
                                # Display the response in the response area
                                with response_area:
                                    st.markdown("### Analysis:")
                                    st.markdown(analysis)
                    else:
                        # Warning message if no features selected
                        st.warning("Please select at least one feature.")
                except Exception as e:
                    # Error message if file processing fails
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please make sure your CSV file is properly formatted.")
            else:
                # Info message if no file uploaded
                st.info("Upload a CSV file to begin your data analysis.")
                
        # Tab 2: ROI Prediction for New Campaign
        with tabs[1]:
            st.header("ROI Prediction for New Campaign")
            
            # Add a checkbox for debug mode
            debug_mode = st.checkbox("Enable debug mode (shows processing details)")
            
            # Check if model has been trained
            if 'best_model' in st.session_state and 'best_model_name' in st.session_state:
                # Show model information
                st.info(f"Using trained model: {st.session_state['best_model_name']}")
                
                # Check if we have the necessary data for prediction
                if ('df' in st.session_state and 'features' in st.session_state and 
                    'categorical_features' in st.session_state):
                    
                    # Display debugging info if enabled
                    if debug_mode:
                        if 'processed_feature_columns' in st.session_state:
                            st.write("Model expects these features:", st.session_state['processed_feature_columns'])
                        st.write("Categorical features:", st.session_state['categorical_features'])
                    
                    st.write("Enter the details of your new marketing campaign to predict ROI:")
                    
                    # Create a form for inputting new campaign features
                    with st.form("prediction_form"):
                        # Get features and their data types from the training data
                        features = st.session_state['features']
                        categorical_features = st.session_state['categorical_features']
                        df = st.session_state['df']
                        target = st.session_state['target']
                        
                        # Dictionary to hold the input values
                        input_values = {}
                        
                        # Create input fields for each feature
                        for feature in features:
                            if feature in categorical_features:
                                # For categorical features, create a dropdown with unique values
                                unique_values = df[feature].dropna().unique().tolist()
                                input_values[feature] = st.selectbox(f"Select {feature}:", unique_values)
                            else:
                                # For numerical features, create a number input
                                # Use default values based on the mean or median of the feature
                                default_value = float(df[feature].median())
                                min_value = float(df[feature].min())
                                max_value = float(df[feature].max())
                                
                                # Create a slider with reasonable step size
                                range_size = max_value - min_value
                                step = range_size / 100.0 if range_size > 0 else 0.1
                                
                                input_values[feature] = st.number_input(
                                    f"Enter {feature}:", 
                                    min_value=min_value,
                                    max_value=max_value,
                                    value=default_value,
                                    step=step
                                )
                        
                        # Submit button
                        submit_button = st.form_submit_button("Predict ROI")
                    
                    # Handle form submission
                    if submit_button:
                        try:
                            # Create a fresh df with all original columns
                            st.write("Preparing input data for prediction...")
                            
                            # Start with just our inputs
                            input_data = pd.DataFrame([input_values])
                            
                            # Display raw input
                            if debug_mode:
                                st.write("Raw input data:")
                                st.write(input_data)
                            
                            # COMPLETELY MANUAL APPROACH FOR PREDICTION
                            # Create an empty dataframe with all the expected columns
                            if 'processed_feature_columns' in st.session_state:
                                expected_columns = st.session_state['processed_feature_columns']
                                
                                # Create an empty dataframe with zeros for all expected columns
                                final_input = pd.DataFrame(0, index=[0], columns=expected_columns)
                                
                                # Now process each feature from our input
                                for feature in features:
                                    if feature in categorical_features:
                                        # Handle categorical by explicit dummy columns
                                        feature_value = input_data[feature].iloc[0]
                                        
                                        # Create the column name that would come from get_dummies
                                        dummy_col = f"{feature}_{feature_value}"
                                        
                                        # If this specific dummy column is expected, set it to 1
                                        if dummy_col in expected_columns:
                                            final_input[dummy_col] = 1
                                            
                                        # Make sure Channel_Display and Channel_Print exist
                                        if "Channel_Display" in expected_columns:
                                            if feature == "Channel" and feature_value == "Display":
                                                final_input["Channel_Display"] = 1
                                            elif "Channel_Display" in final_input.columns:
                                                # Keep it as 0 if it's not Display
                                                pass
                                                
                                        if "Channel_Print" in expected_columns:
                                            if feature == "Channel" and feature_value == "Print":
                                                final_input["Channel_Print"] = 1
                                            elif "Channel_Print" in final_input.columns:
                                                # Keep it as 0 if it's not Print
                                                pass
                                    else:
                                        # For numeric features, just copy the value if the column exists
                                        if feature in expected_columns:
                                            final_input[feature] = input_data[feature].iloc[0]
                                        
                                # Handle var_groupby mapping if needed
                                if ('var_groupby' in st.session_state and 
                                    'location_to_net' in st.session_state and
                                    st.session_state['var_groupby'] in input_data.columns):
                                    
                                    var_groupby = st.session_state['var_groupby']
                                    location_to_net = st.session_state['location_to_net']
                                    
                                    # Map the location value to its mean
                                    location_val = input_data[var_groupby].iloc[0]
                                    if location_val in location_to_net:
                                        mapped_val = location_to_net[location_val]
                                        if var_groupby in final_input.columns:
                                            final_input[var_groupby] = mapped_val
                                
                                # Display final input for debugging
                                if debug_mode:
                                    st.write("Final preprocessed input data:")
                                    st.write(final_input)
                                    
                                # Sanity check: do we have all expected columns?
                                if sorted(final_input.columns.tolist()) == sorted(expected_columns):
                                    st.success("Input data preprocessed successfully, all expected columns present")
                                else:
                                    missing = set(expected_columns) - set(final_input.columns)
                                    extra = set(final_input.columns) - set(expected_columns)
                                    if missing:
                                        st.error(f"Missing columns: {missing}")
                                    if extra:
                                        st.error(f"Extra columns: {extra}")
                                    
                                # Ensure columns are in the exact same order
                                final_input = final_input[expected_columns]
                                
                                # Now make the prediction
                                prediction = st.session_state['best_model'].predict(final_input)[0]
                            
                            # Prepare to preprocess the input data for prediction
                            def preprocess_prediction_data(input_df):
                                # Use the same preprocessing function we defined in the training section
                                # but with is_training=False to ensure consistent feature columns
                                if 'preprocess_data' in locals() or 'preprocess_data' in globals():
                                    return preprocess_data(input_df, is_training=False)
                                else:
                                    # Fallback implementation if the main function isn't available
                                    X = input_df.copy()
                                    
                                    # Handle categorical features using one-hot encoding
                                    if 'categorical_features' in st.session_state:
                                        for col in st.session_state['categorical_features']:
                                            if col in X.columns:
                                                # Create dummies with consistent columns
                                                dummies = pd.get_dummies(X[col], prefix=col)
                                                
                                                # If we have categorical encodings stored
                                                if ('categorical_encodings' in st.session_state and 
                                                    col in st.session_state.categorical_encodings):
                                                    expected_columns = st.session_state.categorical_encodings[col]
                                                    
                                                    # Add missing columns with 0 values
                                                    for dummy_col in expected_columns:
                                                        if dummy_col not in dummies.columns:
                                                            dummies[dummy_col] = 0
                                                    
                                                    # Keep only expected columns
                                                    dummies = dummies[expected_columns]
                                                    
                                                # Add dummy columns to dataframe
                                                X = pd.concat([X, dummies], axis=1)
                                                # Drop original categorical column
                                                X = X.drop(col, axis=1)
                                    
                                    # Map location to its mean target value
                                    if ('var_groupby' in st.session_state and 
                                        'location_to_net' in st.session_state):
                                        var_groupby = st.session_state['var_groupby']
                                        location_to_net = st.session_state['location_to_net']
                                        
                                        if var_groupby in X.columns:
                                            X[var_groupby] = X[var_groupby].map(location_to_net)
                                    
                                    # Ensure all columns from training data are present
                                    if 'processed_feature_columns' in st.session_state:
                                        trained_columns = st.session_state['processed_feature_columns']
                                        
                                        # Find missing and extra columns
                                        missing_cols = set(trained_columns) - set(X.columns)
                                        extra_cols = set(X.columns) - set(trained_columns)
                                        
                                        # Add missing columns with default value 0
                                        for col in missing_cols:
                                            X[col] = 0
                                        
                                        # Remove extra columns
                                        for col in extra_cols:
                                            if col in X.columns:
                                                X = X.drop(col, axis=1)
                                        
                                        # Reorder columns to match training data
                                        X = X[trained_columns]
                                    
                                    return X
                            
                            # Preprocess the input data
                            processed_input = preprocess_prediction_data(input_data)
                            
                            # Make prediction using the trained model
                            prediction = st.session_state['best_model'].predict(processed_input)[0]
                            
                            # Display the prediction
                            st.success(f"Predicted ROI: {prediction:.2f}")
                            
                            # Provide additional context based on the prediction
                            if prediction > 0:
                                st.write("✅ This marketing campaign is predicted to have a positive ROI.")
                                st.write(f"Based on the model, you can expect a return of approximately {prediction:.2f} for this campaign.")
                            else:
                                st.write("⚠️ This marketing campaign is predicted to have a negative or zero ROI.")
                                st.write("You may want to reconsider or adjust the campaign parameters.")
                            
                            # Visualize the prediction with a gauge chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Create a color-coded gauge based on the prediction value
                            if prediction <= 0:
                                color = 'red'
                            elif prediction > 0 and prediction < 5:
                                color = 'orange'
                            else:
                                color = 'green'
                                
                            # Create a simple bar chart representing the ROI
                            ax.barh(["Predicted ROI"], [prediction], color=color)
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add labels
                            for i, v in enumerate([prediction]):
                                if v > 0:
                                    ax.text(v + 0.1, i, f"{v:.2f}", va='center')
                                else:
                                    ax.text(v - 0.1, i, f"{v:.2f}", va='center', ha='right')
                            
                            # Set the x-axis limits based on the prediction
                            max_limit = max(10, prediction * 1.5) if prediction > 0 else 10
                            min_limit = min(-10, prediction * 1.5) if prediction < 0 else -10
                            ax.set_xlim(min_limit, max_limit)
                            
                            # Set the title and labels
                            ax.set_title("Predicted ROI for New Marketing Campaign")
                            ax.set_xlabel("Return on Investment")
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            # Display the chart
                            st.pyplot(fig)
                            
                            # Feature importance interpretation
                            try:
                                # Only display feature importance if the model supports it
                                if hasattr(st.session_state['best_model'], 'feature_importances_'):
                                    st.subheader("Feature Importance")
                                    
                                    # Get feature importances 
                                    importances = st.session_state['best_model'].feature_importances_
                                    
                                    # Create a dataframe for feature importances
                                    if 'processed_feature_columns' in st.session_state:
                                        feature_names = st.session_state['processed_feature_columns']
                                        importance_df = pd.DataFrame({
                                            'Feature': feature_names,
                                            'Importance': importances
                                        })
                                        
                                        # Sort by importance
                                        importance_df = importance_df.sort_values('Importance', ascending=False)
                                        
                                        # Display top 10 features
                                        top_features = importance_df.head(10)
                                        
                                        # Create bar chart
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.barh(top_features['Feature'], top_features['Importance'])
                                        ax.set_title('Top 10 Features by Importance')
                                        ax.set_xlabel('Importance')
                                        plt.tight_layout()
                                        
                                        # Display the chart
                                        st.pyplot(fig)
                                        
                                        # Explain what the most important features mean
                                        st.write("The features listed above have the most influence on the prediction. When planning your campaign, pay special attention to these factors.")
                                else:
                                    st.write("Feature importance is not available for this model type.")
                            except Exception as e:
                                st.warning(f"Could not calculate feature importance: {e}")
                            
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
                            st.write("Please make sure all inputs are valid and try again.")
                else:
                    st.warning("Please analyze your data first in the Data Analysis tab.")
            else:
                st.warning("No trained model available. Please go to the Data Analysis tab and generate visualizations to train a model first.")
            
        # Disclaimer about ML predictions
        st.warning("Disclaimer: ML or AI here cannot promise exact predictions, it is just a tool to get data and algorithm-driven educated predictions.")
    else:
        # Warning if no API key entered
        st.warning("Please enter your OpenAI API key to use this application.")