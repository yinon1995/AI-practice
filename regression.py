import numpy as np
import pandas as pd
import math
# Plots
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn machine learning modules
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error

########################################################
# Exercise 1: HRO in coding frameworks
########################################################
def HRO_basics():
    """Demonstrates basic linear regression workflow"""
    # Initialize linear regression model with intercept term
    model = LinearRegression(fit_intercept=True)
    print(model)  # Display model configuration
    
    # Create input feature array (0-8 in 0.01 increments)
    x = np.arange(0, 8, 0.01)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate target values: y = -1 + 3x + Gaussian noise
    # loc=0: mean noise=0, scale=4: std dev=4, size=len(x): same length as x
    y = -1 + 3*x + np.random.normal(loc=0, scale=4, size=len(x))
    
    # Reshape x to 2D array (required by scikit-learn)
    # -1 means automatic dimension calculation
    model.fit(x.reshape(-1, 1), y)  # Train model
    
    # Calculate predictions and Mean Squared Error (MSE)
    # MSE = average squared difference between predictions and actuals
    mse = metrics.mean_squared_error(y, model.predict(x.reshape(-1, 1)))
    print(f'Model MSE: {mse:.4f}')  # Format to 4 decimal places

def iris_attributes():
    """Explores attributes of the Iris dataset"""
    # Load iris dataset (bunch object: dictionary-like container)
    iris = load_iris()
    
    # Feature matrix (150 samples x 4 features)
    x = iris.data
    # Target vector (150 labels)
    y = iris.target
    
    # Feature names (sepal/petal dimensions)
    feature_names = iris.feature_names
    # Target names (flower species)
    target_names = iris.target_names
    
    # Print object type (Bunch: scikit-learn's custom dataset container)
    print("Type of object iris: ", type(iris))
    
    # Print feature names using list comprehension
    print("Feature names: ")
    [print(f'{i}') for i in feature_names]  # Iterate through names
    
    # Print target names (species)
    print("Target names: ", target_names)
    
    # Print data shapes (x: 150x4 matrix, y: 150-element vector)
    print("\nShape of x and y\n ", x.shape, y.shape)
    
    # Print data types (both NumPy arrays)
    print("\nType of x and y\n ", type(x), type(y))

def tree_regressor():
    """Explores Decision Tree Regressor hyperparameters"""
    # 1. Create default DecisionTreeRegressor
    rtree = DecisionTreeRegressor()
    
    # 2. Print default configuration
    print("Default DecisionTreeRegressor:\n", rtree)
    
    # 3. List all configurable hyperparameters
    print("\nConfigurable hyperparameters and their default values:")
    for k, v in rtree.get_params().items():  # get_params() returns all parameters
        print(f"  {k}: {v}")

    # 4. Summary of key hyperparameters
    """
    Main hyperparameters:
      - criterion:        Split quality measure ('squared_error', 'absolute_error', etc.)
      - splitter:         Split strategy ('best' or 'random')
      - max_depth:        Maximum tree depth (None = unlimited)
      - min_samples_split: Min samples required to split a node
      - min_samples_leaf:  Min samples required at leaf node
      - min_weight_fraction_leaf: Min weighted sample fraction at leaf
      - max_features:     Features considered per split (None = all)
      - random_state:     Randomness control
      - max_leaf_nodes:   Grow tree with max_leaf_nodes
      - min_impurity_decrease: Minimum impurity decrease for split
      - ccp_alpha:        Complexity parameter for pruning
    """
tree_regressor()  # Execute the function

########################################################
# Exercise 4: Predicting abalone age
########################################################
def abaloneRigression():
    """Performs regression on abalone physical measurements"""
    # Load dataset from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    # Read CSV with custom column names
    abalone = pd.read_csv(
        url, 
        sep=',', 
        names=[  # Abalone feature descriptions:
            'sex',              # M/F/I (infant)
            "longest_shell",    # Continuous measurement (mm)
            "diameter",         # Continuous measurement (mm)
            "height",           # Continuous measurement (mm)
            "whole_weight",     # Grams
            "shucked_weight",   # Grams (meat weight)
            "visceral_weight",  # Grams (organ weight)
            "shell_weight",     # Grams
            "rings"             # Target: Age proxy (rings+1.5 = years)
        ]
    )

    # Select key features: shell length and total weight
    abalone = abalone[['longest_shell', 'whole_weight', 'rings']]
    
    # Create scatter plot with ring-based coloring
    plt.grid(True)
    # s=10: point size, cmap='viridis': color map
    plt.scatter(
        abalone.longest_shell, 
        abalone.whole_weight, 
        s=10, 
        c=abalone.rings,  # Color by age rings
        cmap='viridis'     # Color scheme
    )
    plt.colorbar(label='Rings')  # Add color scale legend
    plt.xlabel('Longest shell', size=11)
    plt.ylabel('Whole weight', size=11)
    plt.title('Weight vs shell length for abalone data', size=15)   
    plt.show()  # Display plot

    # Train linear regression model
    x_lm = abalone.iloc[:, 0:2].values  # Feature matrix (columns 0-1)
    y_lm = abalone.rings                # Target vector
    lm = LinearRegression().fit(x_lm, y_lm)  # Train model
    pred_lm = lm.predict(x_lm)          # Generate predictions
    
    # Create prediction vs truth dataframe
    results_dic = {'prediction': pred_lm, 'truth': y_lm}
    results = pd.DataFrame(results_dic)
    print(results.head())  # Show first 5 predictions

    # Create regression plot with uncertainty
    plt.grid(True) 
    # ci=95: 95% confidence interval, s=5: point size
    sns.regplot(
        x=pred_lm, 
        y=y_lm, 
        ci=95,              # Confidence interval
        scatter_kws={'s': 5},  # Scatter point size
        line_kws={"color": "black", 'linewidth': 1}  # Regression line style
    )
    # Add marginal distributions (rug plot)
    sns.rugplot(
        x=pred_lm, 
        y=y_lm, 
        height=0.025,  # Height of rug ticks
        color='k'      # Black color
    )
    plt.title('Truth vs prediction', size=15)
    plt.xlabel('Prediction', size=11)
    plt.ylabel('Truth', size=11)
    plt.show()

    # Calculate Mean Absolute Error (MAE)
    # MAE = average absolute difference between predictions and truths
    mae = mean_absolute_error(pred_lm, y_lm)
    print(f'Mean Absolute Error: {mae:.4f}')

# Key Function Explanations:
# --------------------------
# np.random.normal():
#   Generates normally distributed random numbers
#   loc = mean, scale = standard deviation, size = output shape
#
# reshape(-1, 1):
#   Converts 1D array to 2D column vector (-1 = infer dimension)
#
# sns.regplot():
#   Combines scatterplot and regression line with confidence interval
#
# sns.rugplot():
#   Draws marginal ticks showing data distribution along axes
#
# metrics.mean_squared_error():
#   Computes MSE = average((y_true - y_pred)^2)
#
# mean_absolute_error():
#   Computes MAE = average(|y_true - y_pred|)
#
# get_params():
#   Returns all hyperparameters of a scikit-learn estimator
#
# DecisionTreeRegressor():
#   Regression model that builds decision rules based on features
#   Important parameters control tree complexity and randomness