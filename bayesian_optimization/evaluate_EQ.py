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
        self.d_cond = y_vals
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
