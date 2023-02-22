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
        self.sr_dataset = df

    def eval(self, input_string):
        '''generates score for a given equation'''
        sym_eq = parse_expr(input_string)
        x1_vals = self.sr_dataset['x_1'].values
        x2_vals = self.sr_dataset['x_2'].values
        y_vals = self.sr_dataset['y'].values
        n_rows = self.sr_dataset.shape[0]
        pred_y_list = []
        for i in range(n_rows):
            cur_x1 = x1_vals[i]
            cur_x2 = x2_vals[i]
            pred_y = float(sym_eq.subs({'x_1': cur_x1, 'x_2': cur_x2}))
            pred_y_list.append(pred_y)
        return mean_squared_error(y_vals, pred_y_list)**0.5

if __name__ == '__main__':
    eva = Eval_EQ('../sr_evaluation/sr_dataset_0.csv')
    print(eva.eval('x_1 + x_2'))
