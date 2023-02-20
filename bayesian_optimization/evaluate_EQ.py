import numpy as np
import os
from sympy.parsing.sympy_parser import parse_expr
import pandas as pd
from sklearn.metrics import mean_squared_error

class Eval_EQ(object):
    def __init__(self, sr_dataset_path):
        df = pd.read_csv(sr_dataset_path)
        self.sr_dataset = df

    def eval(self, input_string):
        sym_eq = parse_expr(input_string)
        x1_vals = self.df['x_1'].values
        x2_vals = self.df['x_2'].values
        y_vals = self.df['y'].values
        n_rows = self.df.shape[0]
        pred_y_list = []
        for i in n_rows:
            cur_x1 = x1_vals.iloc[i]
            cur_x2 = x2_vals.iloc[i]
            pred_y = sym_eq.subs({'x_1':cur_x1, 'x_2':cur_x2})
            pred_y_list.append(pred_y)
        return mean_squared_error(y_vals, pred_y_list)
