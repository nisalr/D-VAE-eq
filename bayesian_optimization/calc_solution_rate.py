import sys, os
sys.path.append(os.getcwd())
from bayesian_optimization.symbolic_utils import (clean_pred_model,get_sym_model,round_floats,
                            complexity, rewrite_AIFeynman_model_size)
from sympy import parse_expr, expand_log
import pandas as pd


def solution_match(true_eq, pred_eq):
    pred_eq_simp = expand_log(clean_pred_model(pred_eq), force=True)
    true_eq = expand_log(clean_pred_model(true_eq), force=True)
    print('true pred', true_eq, pred_eq)
    # if the model is somewhat accurate, check and see if it
    # is an exact symbolic match
    sym_diff = round_floats(true_eq - pred_eq_simp)
    sym_frac = round_floats(pred_eq_simp / true_eq)

    if sym_diff.is_constant() or sym_frac.is_constant():
        correct_sol = True
    else:
        correct_sol = False
        sym_err_zero = str(sym_diff) == '0'
        sym_err_const = sym_diff.is_constant()
        sym_frac_const = sym_frac.is_constant()
    return correct_sol


def solution_rate(results_df):
    results_df['is_solution'] = results_df.apply(
        lambda x: solution_match(x['true_eq'], x['pred_eq']), axis=1)
    print(results_df)
    return results_df['is_solution'].mean()



if __name__ == '__main__':
    print(solution_match('x_2*x_1', 'x_2*x_1'))
    print(solution_match('2*x_2*x_1', 'x_2*x_1'))
    df = pd.read_csv('bayesian_optimization/bo_agg_results/EQ_results_34.csv')
    print(solution_rate(df))