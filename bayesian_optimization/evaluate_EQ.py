import numpy as np
import os
from sympy.parsing.sympy_parser import parse_expr
import pandas as pd
from sklearn.metrics import mean_squared_error
import sympy
import time

class Eval_EQ(object):
    def __init__(self, sr_dataset_path):
        df = pd.read_csv(sr_dataset_path)
        sym_eq = parse_expr(df['eq'].iloc[0])
        sym_func = sympy.lambdify(['x_1', 'x_2'], sym_eq)
        x1_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        x2_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        y_vals = np.nan_to_num(sym_func(x1_vals, x2_vals))
        y_mean = np.array([-2.50502381e+04, 2.00070141e+00, 3.47714658e+00,  5.64071075e+00,
                           8.38526966e+00, 1.27348510e+01, 1.88958659e+01, 2.88740704e+01,
                           4.37048087e+01])
        y_std = np.array([1.58219984e+05, 2.66309412e+00, 5.46788612e+00, 1.14458221e+01, 
                          2.28194432e+01, 4.23530231e+01, 7.46783014e+01, 1.25138693e+02,
                          2.00749991e+02])
        self.d_cond = (y_vals - y_mean)/y_std
        self.sr_dataset = df

    def eval(self, input_string):
        '''generates score for a given equation'''
        sym_eq = parse_expr(input_string)
        sym_func = sympy.lambdify(['x_1', 'x_2'], sym_eq)
        x1_vals = self.sr_dataset['x_1'].values
        x2_vals = self.sr_dataset['x_2'].values
        y_vals = self.sr_dataset['y'].values
        pred_y_list = sym_func(x1_vals, x2_vals)
        try:
            return 1/(1 + mean_squared_error(y_vals, pred_y_list)**0.5)
        except ValueError:
            return 0

    def get_dcond(self):
        return self.d_cond

if __name__ == '__main__':
    eva = Eval_EQ('../sr_evaluation/sr_dataset_0.csv')
    print(eva.eval('x_1 + x_2'))
