


# Importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns

# Import additional models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV    

from abess.linear import LinearRegression as AbessLinearRegression # Adaptive Best Subset Selection
import importlib.util
# import feyn # QLattice
# from feyn.qlattice import QLattice # QLattice
                        
import openai
import os
# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets.get("credentials", {}):
            if st.session_state["password"] == st.secrets["credentials"][st.session_state["username"]]:
                st.session_state.authenticated = True
                st.session_state["current_user"] = st.session_state["username"]
                del st.session_state["password"]  # Don't store the password
                return True
            else:
                st.session_state.authenticated = False
                st.error("😕 Password incorrect")
                return False
        else:
            st.session_state.authenticated = False
            st.error("😕 User not found")
            return False

    # First run or if user logs out
    if not st.session_state.authenticated:
        st.title("Data Science Assistant")
        st.markdown("Please enter your username and password")
        
        # Create login form
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                password_entered()

    return st.session_state.authenticated

# Check authentication before showing the main app
if check_password():
    # Set up Streamlit 
    st.title("Data Science Assistant")
    
    # Optional: Show logged in user
    st.sidebar.success(f"Logged in as: {st.session_state.get('current_user', 'User')}")
    
    # Add a logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()
    
    # Check for OpenAI API key in secrets
    if 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
        api_key = st.secrets['openai']['api_key']
        st.success("✅ OpenAI API key configured")
    else:
        # Fall back to manual entry if not in secrets
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if not api_key:
            st.warning("Please enter an OpenAI API key to use this application.")
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Model selection
        model = st.selectbox(
            "Select the model",
            ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"]
        )
        
        # Session management
        session_id = st.text_input("Session ID", value="default_session")
        
        # Initialize session state for storage
        if 'store' not in st.session_state:
            st.session_state.store = {}


    
        # File uploader
        uploaded_file = st.file_uploader("Choose your .csv file", type="csv")
        
        # DataFrame and column handling
        if uploaded_file is not None:
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(uploaded_file)
                
                # Display the column names
                st.write("Columns in the uploaded CSV file:")
                columns = df.columns.tolist()
                for col in columns:
                
                    st.write(col)
                 
                # Feature selection
                ID=st.selectbox("Select the ID variable", columns)
                target = st.selectbox("Select the target variable", columns)
                features_ls = [col for col in columns if col not in [target, ID]]
                
                features = st.multiselect("Select the features", features_ls)
                
                if features:
                    var_convert_range = st.selectbox("Select the variable to convert to quantiles", features)
                    var_groupby = st.selectbox("Select the variable to group by", features)
                    convert_range = st.slider("How many quantiles to convert to", min_value=2, max_value=20, value=10)
                    
                    # Visualization function
                    def generate_visualizations(data, 
                                            test_size=0.3,       # test size
                                            random_state=13,     # random state for reproducibility
                                            var_convert_range=var_convert_range,  # variable to convert to quantiles
                                            var_groupby=var_groupby,  # variable to group by
                                            convert_range=convert_range,  # how many quantiles to convert to
                                            outcome_var=target,  # outcome variable
                                            # Machine Learning Model Parameters:
                                            n_estimators=400,
                                            max_depth=4,
                                            min_samples_split=10,
                                            learning_rate=0.01,
                                            loss='squared_error',
                                            features=features):  # features to use in the model
                        """
                        Generate visualizations for the data
                        """
                        
                        # Create a variable to store positive donors
                        positive_donors = []
                        
                        # Split data into train and test
                        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
                        
                        # --------- Visualization 1: Donation by Income ---------
                        plt.figure(figsize=(12,6))
                        np.random.seed(123)
                        
                        try:
                            # Check if var_convert_range is numerical for qcut
                            if pd.api.types.is_numeric_dtype(train[var_convert_range]):
                                train_with_quantiles = train.assign(
                                    income_quantile=pd.qcut(train[var_convert_range], q=convert_range)
                                )
                                sns.barplot(data=train_with_quantiles,
                                        x="income_quantile", y=outcome_var)
                                plt.title(f"{outcome_var} by {var_convert_range}")
                                plt.xticks(rotation=70)
                                image1 = st.pyplot(plt)
                            else:
                                st.warning(f"{var_convert_range} is not numerical. Cannot create quantiles.")
                                image1 = None
                        except Exception as e:
                            st.error(f"Error in Visualization 1: {e}")
                            image1 = None
                        
                        # --------- Visualization 2: Donation by Location (Historical Data) ---------
                        try:
                            plt.figure(figsize=(12,6))
                            np.random.seed(123)
                            
                            location_donation = train.groupby(var_groupby)[outcome_var].sum()
                            colors = ['blue' if value >= 0 else 'red' for value in location_donation]
                            
                            sns.barplot(data=train, x=var_groupby, y=outcome_var, estimator=sum, ci=None, palette=colors)
                            plt.title(f"{outcome_var} by {var_groupby} - Historical (Train) Data")
                            plt.xticks(rotation=45)
                            image2 = st.pyplot(plt)
                        except Exception as e:
                            st.error(f"Error in Visualization 2: {e}")
                            image2 = None
                        
                        # --------- Identify Locations for Investment ---------
                        try:
                            location_to_net = train.groupby(var_groupby)[outcome_var].mean().to_dict()
                            location_to_invest = {location: net for location, net in location_to_net.items() if net > 0}
                            st.write(f"{var_groupby} to invest in based on historical data: {list(location_to_invest.keys())}")
                            # Handle case where no locations meet criteria
                            if not location_to_invest:
                                st.warning("No locations meet investment criteria.")
                                return image1, image2, None, None, None, []
                                
                            location_policy = test[test[var_groupby].isin(location_to_invest.keys())]
                            
                            # Handle empty dataframe case
                            if location_policy.empty:
                                st.warning("No test data matches the location investment criteria.")
                                return image1, image2, None, None, None, []
                            
                            # --------- Visualization 3: Location Policy on Test Data ---------
                            plt.figure(figsize=(12,6))
                            np.random.seed(123)
                            
                            location_donation = location_policy.groupby(var_groupby)[outcome_var].sum()
                            colors = ['green' if value >= 0 else 'red' for value in location_donation]
                            
                            sns.barplot(data=location_policy, x=var_groupby, y=outcome_var, estimator=sum, ci=None, palette=colors)
                            plt.title(f"{var_groupby} Policy when Applied to Test (Unseen) Data")
                            plt.xticks(rotation=45)
                            image3 = st.pyplot(plt)
                            st.write(f"Potential positive value {var_groupby}s: {location_donation[location_donation > 0].index.tolist()}")
                            # --------- Visualization 4: Histogram of Donations ---------
                            plt.figure(figsize=(10,6))
                            sns.histplot(data=location_policy, x=outcome_var)
                            plt.title(f"Average {outcome_var}: {location_policy[outcome_var].sum() / test.shape[0]:.2f}")
                            image4 = st.pyplot(plt)
                        except Exception as e:
                            st.error(f"Error in Visualizations 3-4: {e}")
                            image3, image4 = None, None
                            return image1, image2, None, None, None, []
                        
                        # --------- Machine Learning Model for Policy Prediction ---------
                        try:
                            
                            
                            def encode(df):
                                return df.copy().assign(**{var_groupby: df[var_groupby].map(location_to_net)})
                            
                            tar = outcome_var
                            
                            # Ensure all features are present
                            missing_features = [f for f in features if f not in train.columns]
                            if missing_features:
                                st.warning(f"Missing features in training data: {missing_features}")
                                return image1, image2, image3, image4, None, []
                                
                            encoded_train = encode(train[features])
                            
                            # Handle missing values in encoded data
                            encoded_train = encoded_train.fillna(encoded_train.mean())
                            
                            # Create a validation set for model selection
                            
                            X_train, X_val, y_train, y_val = train_test_split(
                                encoded_train, train[tar], test_size=0.2, random_state=42
                            )
                            
                            # Progress bar for model evaluation
                            # progress_bar = st.progress(0)
                            # st.write("Evaluating different machine learning models with hyperparameter optimization...")
                            
                            # Define hyperparameter search spaces for each model
                            param_distributions = {
                                "Linear Regression": {},  # No hyperparameters to tune
                                
                                "Ridge Regression": {
                                    'alpha': np.logspace(-3, 3, 20),
                                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                                },
                                
                                "Lasso Regression": {
                                    'alpha': np.logspace(-3, 3, 20),
                                    'selection': ['cyclic', 'random']
                                },
                                
                                "ElasticNet": {
                                    'alpha': np.logspace(-3, 3, 20),
                                    'l1_ratio': np.linspace(0.1, 0.9, 9),
                                    'selection': ['cyclic', 'random']
                                },
                                
                                "Random Forest": {
                                    'n_estimators': [50, 100, 200, 300, 400],
                                    'max_depth': [None, 5, 10, 15, 20, 25],
                                    'min_samples_split': [2, 5, 10, 15],
                                    'min_samples_leaf': [1, 2, 4, 8],
                                    'max_features': ['auto', 'sqrt', 'log2', None]
                                },
                                
                                "AdaBoost": {
                                    'n_estimators': [50, 100, 200, 300],
                                    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
                                    'loss': ['linear', 'square', 'exponential']
                                },
                                
                                # "SVR": {
                                #     'C': np.logspace(-3, 2, 6),
                                #     'gamma': np.logspace(-3, 2, 6),
                                #     'kernel': ['linear', 'poly', 'rbf'],
                                #     'epsilon': [0.01, 0.1, 0.2]
                                # },
                                
                                "KNN": {
                                    'n_neighbors': list(range(1, 31)),
                                    'weights': ['uniform', 'distance'],
                                    'p': [1, 2]  # Manhattan or Euclidean distance
                                },
                                
                                # "Neural Network": {
                                #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                                #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                #     'solver': ['lbfgs', 'sgd', 'adam'],
                                #     'alpha': np.logspace(-5, 3, 5),
                                #     'learning_rate': ['constant', 'invscaling', 'adaptive']
                                # },
                                
                                "Gradient Boosting": {
                                    'n_estimators': [100, 200, 300, 400, 500],
                                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                                    'max_depth': [3, 4, 5, 6, 7, 8],
                                    'min_samples_split': [2, 5, 10, 15, 20],
                                    'min_samples_leaf': [1, 2, 4, 8],
                                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                                    'loss': ['squared_error', 'absolute_error', 'huber']
                                }
                            }
                            
                            # Define base models
                            base_models = {
                                "Linear Regression": LinearRegression(),
                                "Ridge Regression": Ridge(),
                                "Lasso Regression": Lasso(),
                                "ElasticNet": ElasticNet(),
                                "Random Forest": RandomForestRegressor(random_state=42),
                                "AdaBoost": AdaBoostRegressor(random_state=42),
                                #"SVR": SVR(),
                                "KNN": KNeighborsRegressor(),
                                #"Neural Network": MLPRegressor(random_state=42, max_iter=1000),
                                "Gradient Boosting": ensemble.GradientBoostingRegressor(random_state=42)
                            }
                            
                            # Add Abess if available
                            try:
                                
                                base_models["Abess (Best Subset)"] = AbessLinearRegression(fit_intercept=True)
                                param_distributions["Abess (Best Subset)"] = {
                                    'support_size': list(range(1, min(15, len(features)) + 1)),
                                    'ic_type': ['aic', 'bic', 'ebic', 'gic']
                                }
                                #st.info("Abess (Adaptive Best Subset Selection) is available and will be included in model evaluation.")
                            except ImportError:
                                st.warning("Abess package is not installed. To include Abess in model evaluation, install it using: pip install abess")
                                
                            # Store results
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
                            
                            # Create a toggle/checkbox instead of a button
                            #show_hyperparams = st.checkbox("Show Hyperparameter Optimization Details")
                            
                            # Progress bar for model evaluation
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
                                        n_actual_iter = min(n_iter, np.prod([len(v) if hasattr(v, '__len__') else 1 
                                                                            for v in param_space.values()]))
                                        
                                        # If SVR with non-linear kernel, remove gamma for linear kernel
                                        if name == "SVR":
                                            param_grid_svr = param_space.copy()
                                            if 'gamma' in param_grid_svr and 'kernel' in param_grid_svr:
                                                param_grid_svr = {k: (v if k != 'gamma' else ['scale', 'auto'] + list(v)) 
                                                                for k, v in param_grid_svr.items()}
                                            else:
                                                param_grid_svr = param_space
                                            
                                            search = RandomizedSearchCV(model, param_grid_svr, 
                                                                    n_iter=n_actual_iter, 
                                                                    cv=cv, 
                                                                    scoring='r2', 
                                                                    random_state=42,
                                                                    n_jobs=-1)
                                        else:
                                            search = RandomizedSearchCV(model, param_space, 
                                                                    n_iter=n_actual_iter, 
                                                                    cv=cv, 
                                                                    scoring='r2', 
                                                                    random_state=42,
                                                                    n_jobs=-1)
                                        
                                        search.fit(X_train, y_train)
                                        best_model = search.best_estimator_
                                        
                                        # Store the best parameters
                                        st.session_state.hyperparams[name]["has_params"] = True
                                        st.session_state.hyperparams[name]["best_params"] = search.best_params_
                                    
                                    # Evaluate best model
                                    y_train_pred = best_model.predict(X_train)
                                    y_val_pred = best_model.predict(X_val)
                                    
                                    # Calculate metrics
                                    train_r2 = r2_score(y_train, y_train_pred)
                                    val_r2 = r2_score(y_val, y_val_pred)
                                    mse = mean_squared_error(y_val, y_val_pred)
                                    mae = mean_absolute_error(y_val, y_val_pred)
                                    
                                    # Display minimal feedback during processing
                                    #st.write(f"Model: {name} - Validation R²: {val_r2:.4f}")
                                    
                                    # Add to results
                                    results.append({
                                        'Model': name,
                                        'Train R²': train_r2,
                                        'Validation R²': val_r2,
                                        'MSE': mse,
                                        'MAE': mae,
                                        'model_object': best_model
                                    })
                                    
                                except Exception as e:
                                    st.warning(f"Error tuning {name}: {str(e)}")
                            
                            # Clear progress bar
                            progress_bar.empty()
                            
                            # Display hyperparameter details if toggle is checked
                            # if show_hyperparams:
                            #     with hyperparams_container:
                            #         st.subheader("Hyperparameter Optimization Details")
                            #         for name, details in st.session_state.hyperparams.items():
                            #             with st.expander(f"Details for: {name}"):
                            #                 if details.get("no_params", False):
                            #                     st.write(details.get("message", "No hyperparameters to tune"))
                            #                 elif details.get("has_params", False):
                            #                     st.write(f"Best parameters for {name}:")
                            #                     for param, value in details.get("best_params", {}).items():
                            #                         st.write(f"- {param}: {value}")
                            
                            # Create DataFrame of results for display
                            results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model_object'} for r in results])
                            results_df = results_df.sort_values('Validation R²', ascending=False).reset_index(drop=True)
                            
                            # Display results
                            # st.subheader("Model Comparison Results (After Hyperparameter Optimization)")
                            # st.dataframe(results_df.style.format({
                            #     'Train R²': "{:.4f}",
                            #     'Validation R²': "{:.4f}",
                            #     'MSE': "{:.4f}",
                            #     'MAE': "{:.4f}"
                            # }))
                            
                            # Get the best model
                            if not results_df.empty:
                                best_model_idx = results_df['Validation R²'].idxmax()
                                best_model_name = results_df.loc[best_model_idx, 'Model']
                                best_model = [r['model_object'] for r in results if r['Model'] == best_model_name][0]
                                
                                #st.success(f"Best model after optimization: {best_model_name} with Validation R² = {results_df.loc[best_model_idx, 'Validation R²']:.4f}")
                                st.success(f"Best (Optimized) ML model's performance on data unseen to it: R² = {results_df.loc[best_model_idx, 'Validation R²']:.4f}")

                                # Use the best model for final predictions
                                reg = best_model
                                reg.fit(encoded_train, train[tar])
                            else:
                                st.error("No models were successfully trained. Using default Gradient Boosting model.")
                                reg = ensemble.GradientBoostingRegressor(
                                    n_estimators=100, 
                                    max_depth=3, 
                                    learning_rate=0.1, 
                                    random_state=42
                                )
                                reg.fit(encoded_train, train[tar])
                            
                            train_pred = encoded_train.assign(predictions=reg.predict(encoded_train))
                            
                            # Encode test data
                            encoded_test = encode(test[features])
                            encoded_test = encoded_test.fillna(encoded_test.mean())
                            
                            ml_policy = test.assign(prediction=reg.predict(encoded_test))

                            # Extract donors with positive predictions
                            positive_donors_df = ml_policy[ml_policy["prediction"] > 0].copy()
                            
                            # Check if 'donor_id' column exists, otherwise use index
                            if ID in positive_donors_df.columns:
                                id_column = ID
                            else:
                                # Create an ID using the index if no ID column exists
                                positive_donors_df['generated_id'] = positive_donors_df.index
                                id_column = 'generated_id' #assumes that the index is unique/each row has a unique identifier
                            
                            # Calculate the difference between prediction and actual
                            positive_donors_df['diff'] = (positive_donors_df['prediction'] - positive_donors_df[outcome_var])
                            positive_donors_df['diff_percent'] = abs(positive_donors_df['diff'] / positive_donors_df[outcome_var]) * 100
                            
                            # Filter donors where prediction and actual are close (less than 15% difference)
                            # If actual value is 0, we'll include the donor if the prediction is small (< 0.1)
                            close_prediction_donors = positive_donors_df[
                                ((positive_donors_df[outcome_var] != 0) & (positive_donors_df['diff_percent'] < 20)) | 
                                ((positive_donors_df[outcome_var] == 0) & (positive_donors_df['prediction'] < 0.1))
                            ].copy()
                            
                            # Create a list of dictionaries with donor info
                            positive_donors = close_prediction_donors[[id_column
                                                                    #, outcome_var
                                                                    , 'prediction'
                                                                    #, 'diff'
                                                                    , 'diff_percent'
                                                                    ]].to_dict('records')
                            
                            # Add a rank based on closeness of prediction (smaller difference is better)
                            positive_donors_sorted = sorted(positive_donors, key=lambda x: x['diff_percent'])
                            for rank, donor in enumerate(positive_donors_sorted, 1):
                                donor['rank'] = rank
                            
                            
                            train_r2 = r2_score(y_true=train[tar], y_pred=train_pred["predictions"])
                            test_r2 = r2_score(y_true=test[tar], y_pred=reg.predict(encoded_test))
                            
                            st.write(f"Train R²: {train_r2:.4f}")
                            st.write(f"Test R²: {test_r2:.4f}")
                            
                            # --------- Visualization 5: ML vs. Location-Based Policy ---------
                            plt.figure(figsize=(10,6))
                            ml_df = ml_policy[ml_policy["prediction"] > 0]
                            locations_df = ml_policy[ml_policy[var_groupby].isin(location_to_invest.keys())]
                            
                            # Handle empty dataframes
                            if ml_df.empty:
                                st.warning("ML policy produced no positive predictions.")
                                return image1, image2, image3, image4, None, []
                                
                            sns.histplot(data=ml_df, x=outcome_var, color="C1", label="ML Policy")
                            sns.histplot(data=locations_df, x=outcome_var, label=f"{var_groupby} Policy")
                            
                            plt.title(f"ML/AI {outcome_var}: {ml_df[outcome_var].sum() / test.shape[0]:.2f} | "
                                    f"{var_groupby} Policy {outcome_var}: {locations_df[outcome_var].sum() / test.shape[0]:.2f}")
                            
                            plt.legend()
                            image5 = st.pyplot(plt)
                            
                            return image1, image2, image3, image4, image5, positive_donors
                        except Exception as e:
                            st.error(f"Error in Machine Learning Model: {e}")
                            return image1, image2, image3, image4, None, []
                    
                    # Button to generate visualizations
                    if st.button("Generate Visualizations"):
                        with st.spinner("Generating...."):
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

                            # Store positive donors in session state
                            if positive_donors:
                                st.session_state.positive_donors = positive_donors
                                st.success(f"Found {len(positive_donors)} {ID}s with difference between prediction and actual {target} < 20%")
                                
                                # Show top 10 donors
                                st.subheader("Top 10 Donors with Most Accurate Predictions")
                                top_donors_df = pd.DataFrame(positive_donors[:10])
                                # Format percentage columns for better readability
                                if 'diff_percent' in top_donors_df.columns:
                                    top_donors_df['diff_percent'] = top_donors_df['diff_percent'].map('{:.2f}%'.format)
                                st.dataframe(top_donors_df)
                                
                                # Add download button for full list
                                full_donors_df = pd.DataFrame(positive_donors)
                                if 'diff_percent' in full_donors_df.columns:
                                    full_donors_df['diff_percent'] = full_donors_df['diff_percent'].map(lambda x: '{:.2f}'.format(x))
                                csv = full_donors_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    f"Download this table as CSV (Please see disclaimer below)",
                                    csv,
                                    "{id_column}_list_with_predicted_{outcome_var}.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                            # Save the visualizations for LLM analysis (if that function exists)
                            if 'save_visualizations' in locals() or 'save_visualizations' in globals():
                                save_visualizations(i1, i2, i3, i4, i5)
                    
                    # GPT Analysis section
                    st.header("Chat Bot")
                    st.info("Ask questions about the data and get AI-powered insights")
                    response_area = st.container()
                    def generate_analysis(user_query, data):
                        try:
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

                            # Add positive donors information if available
                            positive_donors_info = ""
                            if 'positive_donors' in st.session_state and st.session_state.positive_donors:
                                donors = st.session_state.positive_donors
                                try:
                                    positive_donors_info = f"""
                                    \n\nAccurate prediction donors summary:
                                    - Total donors with accurate predictions: {len(donors)}
                                    - Average predicted donation: {sum(d['prediction'] for d in donors) / len(donors):.2f}
                                    """
                                    
                                    # Only add these fields if they exist in all donors
                                    if all('diff_percent' in d for d in donors):
                                        positive_donors_info += f"- Average difference percentage: {sum(float(d['diff_percent']) for d in donors) / len(donors):.2f}%\n"
                                    
                                    positive_donors_info += "\nTop 5 donors with relatively more accurate predictions:"
                                    
                                    for i, donor in enumerate(donors, 1):
                                        # Find the ID field more safely
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
                            # Add ML rationale context if keyword is detected in the query
                            ml_rationale = ""
                            if any(keyword in user_query.lower() for keyword in ["why", "better", "rationale", "explain", "compared", "advantage", "ml policy", "machine learning"]):
                                ml_rationale = f"""
                                \n\nRATIONALE FOR ML PERFORMANCE:
                                
                                1. {var_groupby} policy uses {ID} location as the only deciding factor. The policy doesn't consider other important details about individual {id}s.
                                
                                2. ML-based policy considers {ID} behavior, previous {target} history, {features} and {var_groupby}n all together - identifies complex patterns and interactions between these features.
                                
                                3. ML builds a more nuanced understanding of what makes someone a "high net {ID}."
                                
                                4. The ML algorithm used here can detect subtle, nonlinear relationships between {features_ls}.
                                
                                5. Empirical evidence: In testing, ML Policy achieved a net {target} average {sum(d['prediction'] for d in donors) / len(donors):.2f} compared to {var_groupby} Policy's {locations_df[outcome_var].sum() / test.shape[0]:.2f}.

                        
                                """
                                        
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "You are a data scientist who analyzes data. Provide insights, patterns, and recommendations based on the data."},
                                    {"role": "user", "content": f"Dataset context:\n{data_summary}{positive_donors_info}{ml_rationale}\n\nUser question: {user_query}"}
                                ]
                            )
                            return response.choices[0].message.content
                        except Exception as e:
                            return f"Error generating analysis: {str(e)}"
                    
                    # User input for analysis
                    # Create a placeholder for responses
                    
                    user_query = st.text_input("Ask a question about your data:")
                    
                    if user_query:
                        with st.spinner("Analyzing data..."):
                            analysis = generate_analysis(user_query, df)
                            with response_area:
                                st.markdown("### Analysis:")
                                st.markdown(analysis)
                else:
                    st.warning("Please select at least one feature.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please make sure your CSV file is properly formatted.")
        else:
            st.info("Upload a CSV file to begin your data analysis.")
            
        st.warning("Disclaimer: ML or AI here cannot promise exact predictions, it is just a tool to get data and algorithm-driven educated predictions.")
    else:
        st.warning("Please enter your OpenAI API key to use this application.")
    # code file path: cd '/Users/abeautifulmind/Dropbox/untitled folder/code/'