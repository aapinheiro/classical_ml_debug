import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class MachineLearningModelDebug:
    def __init__(self, model_type='classification'):
        self.model_type = model_type
        if self.model_type == 'classification':
            self.model = LogisticRegression()
        elif self.model_type == 'regression':
            self.model = LinearRegression()
        else:
            raise ValueError("Invalid model type. Choose 'classification' or 'regression'.")

    def learning_curve(self, X, y, train_sizes, test_size=0.2, random_state=None, desired_performance=None, mean_error_lines=True):
        train_errors = []
        test_errors = []

        for size in train_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, test_size=test_size, random_state=random_state)

            self.model.fit(X_train, y_train)

            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)

            if self.model_type == 'classification':
                train_error = log_loss(y_train, self.model.predict_proba(X_train))
                test_error = log_loss(y_test, self.model.predict_proba(X_test))
                error = 'log-loss'
            elif self.model_type == 'regression':
                train_error = mean_squared_error(y_train, y_train_pred)
                test_error = mean_squared_error(y_test, y_test_pred)
                error = 'mse'

            train_errors.append(train_error)
            test_errors.append(test_error)

        # Calculate mean of training and test errors
        mean_train_error = np.mean(train_errors)
        mean_test_error = np.mean(test_errors)

        # Printing the errors
        print(f"The mean of train error: {mean_train_error}")
        print(f"The mean of test error: {mean_test_error}")

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_errors, label='Training Loss')
        plt.plot(train_sizes, test_errors, label='Test Loss')
        
        if mean_error_lines: 
          # Plot horizontal lines for mean errors
          plt.axhline(y=mean_train_error, color='b', linestyle='--', label='Mean Training Loss')
          plt.axhline(y=mean_test_error, color='g', linestyle='--', label='Mean Test Loss')
        
        else: 
          pass
        
        if desired_performance is not None:
            plt.axhline(y=desired_performance, color='r', linestyle='--', label='Desired Performance')

        plt.title('Learning Curve')
        plt.xlabel('Training Set Size')
        plt.ylabel(f'Error ({error})')
        plt.legend()
        plt.grid(True)
        plt.show()
