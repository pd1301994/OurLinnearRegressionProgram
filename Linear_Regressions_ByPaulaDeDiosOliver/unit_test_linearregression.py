import pandas as pd
import numpy as np
import unittest
from sklearn import linear_model

data = pd.read_csv('test.csv')
x = data['x'].values.reshape(-1, 1)
y= data['y'].values.reshape(-1, 1)
ols = linear_model.LinearRegression()
ols.fit(x, y)
b0 = float(ols.intercept_)
b1 = float(ols.coef_)
coef_string = f'{b0}+{b1}'
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    return f'{b_0}+{b1}'

class TestString (unittest.TestCase):
    def test_coeficients (self):
        self.assertEqual(estimate_coef(x,y),coef_string)









