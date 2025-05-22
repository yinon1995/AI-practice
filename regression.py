import numpy as np
import pandas as pd
import math
# plots
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error


def HRO_basics():
    model = LinearRegression(fit_intercept=True)
    print(model)
    x = np.arange(0, 8, 0.01)
    print(x)




