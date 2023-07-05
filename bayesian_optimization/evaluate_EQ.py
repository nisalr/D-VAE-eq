import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import os
from sympy.parsing.sympy_parser import parse_expr
import pandas as pd
from sklearn.metrics import mean_squared_error
import sympy
import time
from src.nesymres.architectures.model import Model
from src.nesymres.dclasses import FitParams, NNEquation, BFGSParams
from functools import partial
import torch
import json
import omegaconf
import pickle

class Eval_EQ(object):
    def __init__(self, sr_dataset_path, embed_mode=None, res_path=None):
        df = pd.read_csv(sr_dataset_path)
        if embed_mode == 'simple':
            with open(res_path + '/ycond_mean.pkl', 'rb') as f:
                ycond_mean = pickle.load(f)
            with open(res_path + '/ycond_std.pkl', 'rb') as f:
                ycond_std = pickle.load(f)
            sym_eq = parse_expr(df['eq'].iloc[0])
            sym_func = sympy.lambdify(['x_1', 'x_2'], sym_eq)
            x1_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            x2_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            y_vals = np.nan_to_num(sym_func(x1_vals, x2_vals))
            y_vals = (y_vals - ycond_mean) / ycond_std
            self.d_cond = y_vals
        elif embed_mode == 'nesymres':
            with open(res_path + '/ycond_mean.pkl', 'rb') as f:
                ycond_mean = pickle.load(f)
            with open(res_path + '/ycond_std.pkl', 'rb') as f:
                ycond_std = pickle.load(f)
            ## Load equation configuration and architecture configuration
            with open('src/nesymres/eq_setting.json', 'r') as json_file:
              eq_setting = json.load(json_file)

            cfg = omegaconf.OmegaConf.load("src/nesymres/config.yaml")
            weights_path = "weights/nesymres_100M.ckpt"
            model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
                ## Set up BFGS load rom the hydra config yaml

            bfgs = BFGSParams(
                    activated=cfg.inference.bfgs.activated,
                    n_restarts=cfg.inference.bfgs.n_restarts,
                    add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
                    normalization_o=cfg.inference.bfgs.normalization_o,
                    idx_remove=cfg.inference.bfgs.idx_remove,
                    normalization_type=cfg.inference.bfgs.normalization_type,
                    stop_time=cfg.inference.bfgs.stop_time,
                )
            params_fit = FitParams(word2id=eq_setting["word2id"],
                                   id2word={int(k): v for k,v in eq_setting["id2word"].items()},
                                   una_ops=eq_setting["una_ops"],
                                   bin_ops=eq_setting["bin_ops"],
                                   total_variables=list(eq_setting["total_variables"]),
                                   total_coefficients=list(eq_setting["total_coefficients"]),
                                   rewrite_functions=list(eq_setting["rewrite_functions"]),
                                   bfgs=bfgs,
                                   beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                    )
            embed_func = partial(model.dataset_encoding, cfg_params=params_fit)
            df['x_3'] = 0
            x_vals = np.expand_dims(df[['x_1', 'x_2', 'x_3']].values, 0)
            y_vals = np.expand_dims(df['y'].values, 0)
            self.d_cond = embed_func(x_vals, y_vals).mean(axis=2).reshape(-1).detach().cpu().numpy()
            self.d_cond = (self.d_cond - ycond_mean) / ycond_std
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
