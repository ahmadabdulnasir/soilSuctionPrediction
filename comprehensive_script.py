import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Load and explore the dataset
def load_and_explore_data():
    # Load the CSV data
    df = pd.read_csv('abdullahi_karaye_new_dataset.csv')
    
    print("=== DATASET OVERVIEW ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    
    print("\n=== STATISTICAL SUMMARY ===")
    print(df.describe())
    
    return df

# Data preprocessing
def preprocess_data(df):
    print("\n=== DATA PREPROCESSING ===")
    
    # Check for outliers using IQR method
    print("Checking for outliers...")
    outliers_info = {}
    
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        outliers_info[column] = len(outliers)
    
    print("Outliers per column:")
    for col, count in outliers_info.items():
        print(f"{col}: {count} outliers")
    
    # Handle scientific notation in 'k' column
    df['k'] = df['k'].astype(float)
    
    # Create log transformation for highly skewed 'k' values
    df['log_k'] = np.log10(df['k'].abs() + 1e-15)  # Add small constant to avoid log(0)
    
    print(f"\nProcessed dataset shape: {df.shape}")
    return df

# Exploratory Data Analysis
def exploratory_analysis(df):
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    
    # Correlation analysis
    features = ['pd', 'w', 'pb', 'P.I', 'O', 'k', 'n', 'D60', 'Ya']
    correlation_with_target = []
    
    for feature in features:
        corr = df[feature].corr(df['Suction'])
        correlation_with_target.append({
            'Feature': feature,
            'Correlation': corr,
            'Abs_Correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlation_with_target)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    print("Feature correlation with Suction (sorted by absolute value):")
    print(corr_df)
    
    # Feature importance ranking
    top_features = corr_df.head(6)['Feature'].tolist()
    print(f"\nTop 6 features by correlation: {top_features}")
    
    return corr_df, top_features

# Feature engineering
def feature_engineering(df):
    print("\n=== FEATURE ENGINEERING ===")
    
    # Create interaction features
    df['pd_w_ratio'] = df['pd'] / (df['w'] + 1e-6)  # Avoid division by zero
    df['pb_pd_ratio'] = df['pb'] / df['pd']
    df['w_O_ratio'] = df['w'] / (df['O'] + 1e-6)
    
    # Polynomial features for top correlated variables
    df['pd_squared'] = df['pd'] ** 2
    df['w_squared'] = df['w'] ** 2
    
    # Create categorical features based on ranges
    df['moisture_level'] = pd.cut(df['w'], bins=3, labels=['Low', 'Medium', 'High'])
    df['density_level'] = pd.cut(df['pd'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Convert categorical to numeric and store thresholds for prediction
    df['moisture_level_num'] = df['moisture_level'].cat.codes
    df['density_level_num'] = df['density_level'].cat.codes
    
    # Store thresholds for use in prediction (global variables for simplicity)
    global MOISTURE_THRESHOLDS, DENSITY_THRESHOLDS
    MOISTURE_THRESHOLDS = df['w'].quantile([0.33, 0.67]).values
    DENSITY_THRESHOLDS = df['pd'].quantile([0.33, 0.67]).values
    
    print("New engineered features created:")
    new_features = ['pd_w_ratio', 'pb_pd_ratio', 'w_O_ratio', 'pd_squared', 'w_squared', 
                   'moisture_level_num', 'density_level_num', 'log_k']
    print(new_features)
    print(f"Moisture thresholds: {MOISTURE_THRESHOLDS}")
    print(f"Density thresholds: {DENSITY_THRESHOLDS}")
    
    return df, new_features

# Model training and evaluation
def train_models(df, top_features, engineered_features):
    print("\n=== MODEL TRAINING ===")
    
    # Prepare features and target
    all_features = top_features + engineered_features
    X = df[all_features].fillna(df[all_features].mean())  # Handle any NaN values
    y = df['Suction']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for linear models and SVR
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Train model
        model.fit(X_train_model, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_model)
        y_pred_test = model.predict(X_test_model)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'Train R2': train_r2,
            'Test R2': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'CV Mean R2': cv_scores.mean(),
            'CV Std R2': cv_scores.std(),
            'Model': model,
            'Predictions': y_pred_test
        }
    
    return results, X_test, y_test, scaler, all_features

# Model comparison and selection
def compare_models(results):
    print("\n=== MODEL COMPARISON ===")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train R2': [results[model]['Train R2'] for model in results.keys()],
        'Test R2': [results[model]['Test R2'] for model in results.keys()],
        'Test RMSE': [results[model]['Test RMSE'] for model in results.keys()],
        'Test MAE': [results[model]['Test MAE'] for model in results.keys()],
        'CV Mean R2': [results[model]['CV Mean R2'] for model in results.keys()],
        'CV Std R2': [results[model]['CV Std R2'] for model in results.keys()]
    })
    
    print(comparison_df.round(4))
    
    # Select best model based on CV performance and generalization
    best_model_name = comparison_df.loc[comparison_df['CV Mean R2'].idxmax(), 'Model']
    best_model = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model CV R2: {best_model['CV Mean R2']:.4f} ± {best_model['CV Std R2']:.4f}")
    print(f"Best Model Test R2: {best_model['Test R2']:.4f}")
    print(f"Best Model Test RMSE: {best_model['Test RMSE']:.4f}")
    
    return best_model_name, best_model, comparison_df

# Feature importance analysis
def analyze_feature_importance(best_model_name, best_model, feature_names):
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ({best_model_name}) ===")
    
    if best_model_name == 'Random Forest':
        importances = best_model['Model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Feature Importance (Random Forest):")
        print(feature_importance_df)
        
    elif best_model_name in ['Ridge Regression', 'Lasso Regression']:
        coefficients = best_model['Model'].coef_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(f"Feature Coefficients ({best_model_name}):")
        print(feature_importance_df)
    
    return feature_importance_df

# Feature engineering function (reusable)
def engineer_features(input_data):
    """
    Generate engineered features from original features
    
    Parameters:
    - input_data: dictionary with original features
    
    Returns:
    - dictionary with all features (original + engineered)
    """
    # Copy original features
    features = input_data.copy()
    
    # Create interaction features
    features['pd_w_ratio'] = features['pd'] / (features['w'] + 1e-6)  # Avoid division by zero
    features['pb_pd_ratio'] = features['pb'] / features['pd']
    features['w_O_ratio'] = features['w'] / (features['O'] + 1e-6)
    
    # Polynomial features for top correlated variables
    features['pd_squared'] = features['pd'] ** 2
    features['w_squared'] = features['w'] ** 2
    
    # Log transformation for permeability
    features['log_k'] = np.log10(features['k'] + 1e-15)  # Add small constant to avoid log(0)
    
    # Create categorical features based on typical ranges (from training data analysis)
    # These thresholds should be saved from training phase
    if features['w'] <= 14.5:
        features['moisture_level_num'] = 0  # Low
    elif features['w'] <= 18.5:
        features['moisture_level_num'] = 1  # Medium
    else:
        features['moisture_level_num'] = 2  # High
    
    if features['pd'] <= 1.60:
        features['density_level_num'] = 0  # Low
    elif features['pd'] <= 1.70:
        features['density_level_num'] = 1  # Medium
    else:
        features['density_level_num'] = 2  # High
    
    return features

# Prediction function
def make_prediction(model, scaler, feature_names, original_input):
    """
    Make a prediction for new input values using only original features
    
    Parameters:
    - model: trained model
    - scaler: fitted scaler (None if not needed)
    - feature_names: list of all feature names (original + engineered)
    - original_input: dictionary with ONLY original features
                     Expected keys: ['pd', 'w', 'pb', 'P.I', 'O', 'k', 'n', 'D60', 'Ya']
    
    Returns:
    - predicted suction value
    """
    # Validate input
    required_features = ['pd', 'w', 'pb', 'P.I', 'O', 'k', 'n', 'D60', 'Ya']
    missing_features = [f for f in required_features if f not in original_input]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Generate all features (original + engineered)
    all_features = engineer_features(original_input)
    
    # Create input array in the correct order
    input_array = np.array([all_features[feature] for feature in feature_names]).reshape(1, -1)
    
    # Scale if needed (for linear models and SVR)
    if scaler is not None:
        if hasattr(model, 'kernel') or hasattr(model, 'coef_'):  # SVR or Linear models
            input_array = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    
    return prediction

# Updated prediction class for production use
class SuctionPredictor:
    """
    Production-ready suction prediction class
    """
    def __init__(self, model, scaler, feature_names, model_name):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_name = model_name
        self.required_features = ['pd', 'w', 'pb', 'P.I', 'O', 'k', 'n', 'D60', 'Ya']
    
    def predict(self, input_data):
        """
        Predict suction from original features only
        
        Parameters:
        - input_data: dict with keys ['pd', 'w', 'pb', 'P.I', 'O', 'k', 'n', 'D60', 'Ya']
        
        Returns:
        - predicted suction value
        """
        return make_prediction(self.model, self.scaler, self.feature_names, input_data)
    
    def predict_batch(self, input_list):
        """
        Predict suction for multiple inputs
        
        Parameters:
        - input_list: list of dictionaries, each with original features
        
        Returns:
        - list of predictions
        """
        return [self.predict(input_data) for input_data in input_list]
    
    def get_feature_info(self):
        """
        Return information about required input features
        """
        feature_info = {
            'pd': 'Dry density (g/cm³)',
            'w': 'Water content (%)',
            'pb': 'Bulk density (g/cm³)',
            'P.I': 'Plasticity Index',
            'O': 'Optimum moisture content',
            'k': 'Permeability coefficient (m/s)',
            'n': 'Porosity',
            'D60': 'Particle size parameter (mm)',
            'Ya': 'Apparent cohesion (kPa)'
        }
        return feature_info

# Main execution function
def main():
    print("SUCTION PREDICTION MODEL DEVELOPMENT")
    print("=" * 50)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    # Step 2: Preprocess data
    df = preprocess_data(df)
    
    # Step 3: Exploratory analysis
    corr_df, top_features = exploratory_analysis(df)
    
    # Step 4: Feature engineering
    df, engineered_features = feature_engineering(df)
    
    # Step 5: Train models
    results, X_test, y_test, scaler, all_features = train_models(df, top_features, engineered_features)
    
    # Step 6: Compare and select best model
    best_model_name, best_model, comparison_df = compare_models(results)
    
    # Step 7: Feature importance analysis
    feature_importance_df = analyze_feature_importance(best_model_name, best_model, all_features)
    
    print("\n=== MODEL DEVELOPMENT COMPLETE ===")
    print(f"Best performing model: {best_model_name}")
    print(f"Model ready for deployment and predictions!")
    
    # Example prediction
    print("\n=== EXAMPLE PREDICTION ===")
    example_input = {
        'pd': 1.60,
        'w': 18.0,
        'pb': 30.0,
        'P.I': 20.0,
        'O': 0.30,
        'k': 1e-10,
        'n': 0.35,
        'D60': 0.10,
        'Ya': 40,
        'pd_w_ratio': 1.60/18.0,
        'pb_pd_ratio': 30.0/1.60,
        'w_O_ratio': 18.0/0.30,
        'pd_squared': 1.60**2,
        'w_squared': 18.0**2,
        'moisture_level_num': 1,
        'density_level_num': 1,
        'log_k': np.log10(1e-10 + 1e-15)
    }
    
    if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
        prediction = make_prediction(best_model['Model'], scaler, all_features, example_input)
    else:
        input_array = np.array([example_input[feature] for feature in all_features]).reshape(1, -1)
        prediction = best_model['Model'].predict(input_array)[0]
    
    print(f"Predicted Suction: {prediction:.2f}")
    
    return df, best_model, scaler, all_features, comparison_df

# Run the analysis
if __name__ == "__main__":
    df, best_model, scaler, features, comparison = main()
