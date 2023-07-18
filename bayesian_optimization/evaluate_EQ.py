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
from src.nesymres.architectures.data import tokenize, de_tokenize
from src.nesymres.architectures.bfgs import bfgs
from dataset.data_gen import Generator

from functools import partial
import torch
import json
import omegaconf
import pickle

from sympy import Symbol, simplify, factor, Float, preorder_traversal, Integer


def round_sympy_expr(sym_expr, round_digits=3):
    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            if int(round(a, round_digits)) == round(a, round_digits):
                sym_expr = sym_expr.subs(a, int(round(a, round_digits)))
            else:
                sym_expr = sym_expr.subs(a, round(a, round_digits))
    return sym_expr



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

            bfgs_params = BFGSParams(
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
                                   bfgs=bfgs_params,
                                   beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                    )
            self.params_fit = params_fit
            embed_func = partial(model.dataset_encoding, cfg_params=params_fit)
            df['x_3'] = 0
            x_vals = np.expand_dims(df[['x_1', 'x_2', 'x_3']].values, 0)
            y_vals = np.expand_dims(df['y'].values, 0)
            self.d_cond = embed_func(x_vals, y_vals).mean(axis=2).reshape(-1).detach().cpu().numpy()
            self.d_cond = (self.d_cond - ycond_mean) / ycond_std
        self.sr_dataset = df

    def eval(self, input_string, samp_count=500):
        '''generates score for a given equation'''
        sym_eq = parse_expr(input_string)
        sampled_df = self.sr_dataset.sample(samp_count)
        # sym_func = sympy.lambdify(['x_1', 'x_2'], sym_eq)
        # x1_vals = self.sr_dataset['x_1'].values
        # x2_vals = self.sr_dataset['x_2'].values
        # y_vals = self.sr_dataset['y'].values
        # pred_y_list = sym_func(x1_vals, x2_vals)
        x_vals = torch.FloatTensor(sampled_df[['x_1', 'x_2', 'x_3']].values).unsqueeze(0)
        y_vals = torch.FloatTensor(sampled_df[['y']].values).unsqueeze(0)

        prefix_expr = Generator.sympy_to_prefix(sym_eq)
        token_expr = torch.Tensor(tokenize(prefix_expr, self.params_fit.word2id))
        print(token_expr)
        func_best, const_best, mse_loss, fitted_expr = bfgs(token_expr, x_vals, y_vals, self.params_fit)
        print('printing bfgs results', func_best, const_best, mse_loss, fitted_expr)
        try:
            score = 1/(1 + mse_loss**0.5)
            if np.isnan(score):
                return 0, round_sympy_expr(func_best)
            return score, round_sympy_expr(func_best)
        except ValueError:
            return 0, round_sympy_expr(func_best)

    def get_dcond(self):
        return self.d_cond

if __name__ == '__main__':
    eva = Eval_EQ('../sr_evaluation/sr_dataset_0.csv')
    print(eva.eval('x_1 + x_2'))
