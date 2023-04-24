import csv

import numpy as np
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_ideal = pd.read_csv('ideal.csv')
list_of_column_names = []
ols = linear_model.LinearRegression()
mpl.use('TkAgg')
class RegressionFunction:
    def linearregression(self, data):
        best_functions = []
        try:
            i = 0
            # We asssume there is always going to be an x column, if not data is just not good reshape
            if data.columns.values[0] == 'x':
                x = data['x'].values.reshape(-1, 1)
            else:
                return 'The name of the columns is  incorrect, try to correct them'
            better_score = []
            for row in data:
                if i == 0:
                    i = i + 1
                    continue
                y = data[row].values.reshape(-1, 1)
                ols.fit(x, y)
                # choosing the best fit out of the functions
                better_score.append((ols.score(x, y), row))
            better_score.sort(reverse=True)
            best_functions = []
            for elements in better_score[:4]:
                y = data[elements[1]].values.reshape(-1, 1)
                ols.fit(x, y)
                # print(f'f(x) = {ols.intercept_}+{ols.coef_}x')
                best_functions.append((elements[1], float(ols.intercept_), float(ols.coef_)))
        except FileNotFoundError:
            print("File not found")
        except ValueError:
            print("Invalid values")
        return best_functions


class ManagingData(RegressionFunction):
    def __init__(self):
        RegressionFunction.__init__(self)
        self.x_ideal = data_ideal['x'].values.reshape(-1, 1)

    def plotting_functions(self, x_scatter, y_scatter, x_regression, b0, b1, title):
        plt.scatter(x_scatter, y_scatter, color='blue')
        plt.plot(x_regression, [b0 + b1 * float(i) for i in x_regression], color='red')
        plt.title(title)
        plt.show()


    def saving_functions_in_csv (self, file_name, data):
        file = f'{file_name}.csv'
        header = []
        rest = []
        for element in data:
            header.append(element[0])
            rest.append(f'{element[1]} + {element[2]}*x')

        with open(file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(rest)


lr = ManagingData()
processed_ideal_data = lr.linearregression(data_ideal)
print (processed_ideal_data)


