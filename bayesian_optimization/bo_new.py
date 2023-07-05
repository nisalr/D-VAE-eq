import pdb
import pickle
import sys
import os
import os.path
import collections
import torch
from tqdm import tqdm
import itertools
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy.stats import pearsonr
# sys.path.append('%s/../software/enas' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, '../')
from models import *
from util import *
from bayesian_optimization.evaluate_EQ import Eval_EQ
from shutil import copy
from bayes_opt import BayesianOptimization, UtilityFunction
from scipy.optimize import dual_annealing
import time


'''Experiment settings'''
parser = argparse.ArgumentParser(description='Bayesian optimization experiments.')
# must specify
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--save-appendix', default='',
                    help='what is appended to data-name as save-name for results')
parser.add_argument('--checkpoint', type=int, default=300,
                    help="load which epoch's model checkpoint")
parser.add_argument('--res-dir', default='res/',
                    help='where to save the Bayesian optimization results')
# BO settings
parser.add_argument('--predictor', action='store_true', default=False,
                    help='if True, use the performance predictor instead of SGP')
parser.add_argument('--grad-ascent', action='store_true', default=False,
                    help='if True and predictor=True, perform gradient-ascent with predictor')
parser.add_argument('--BO-rounds', type=int, default=10,
                    help="how many rounds of BO to perform")
parser.add_argument('--BO-batch-size', type=int, default=50,
                    help="how many data points to select in each BO round")
parser.add_argument('--sample-dist', default='uniform',
                    help='from which distrbiution to sample random points in the latent \
                    space as candidates to select; uniform or normal')
parser.add_argument('--random-baseline', action='store_true', default=False,
                    help='whether to include a baseline that randomly selects points \
                    to compare with Bayesian optimization')
parser.add_argument('--random-as-train', action='store_true', default=False,
                    help='if true, no longer use original train data to initialize SGP \
                    but randomly generates 1000 initial points as train data')
parser.add_argument('--random-as-test', action='store_true', default=False,
                    help='if true, randomly generates 100 points from the latent space \
                    as the additional testing data')
parser.add_argument('--vis-2d', action='store_true', default=False,
                    help='do visualization experiments on 2D space')
parser.add_argument('--dnum', type=int, default=0,
                    help='number of SR dataset to evaluate')

# can be inferred from the cmd_input.txt file, no need to specify
parser.add_argument('--data-type', default='ENAS',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--model', default='DVAE', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
parser.add_argument('--cond', action='store_true', default=False,
                    help='condition DVAE on dataset')
parser.add_argument('--cond-size', type=int, default=None,
                    help='size of the dataset embedding vector')
parser.add_argument('--optimizer', default='bayes', help='black box optimizer to use: bayes, \
                    anneal')

args = parser.parse_args()
is_cond = args.cond
cond_size = args.cond_size
dataset_num = args.dnum
data_name = args.data_name
save_appendix = args.save_appendix
res_dir = args.res_dir
data_dir = 'results/{}_{}/'.format(data_name, save_appendix)  # data and model folder
save_dir = 'bayesian_optimization/{}results_{}/'.format(res_dir, save_appendix)  # where to save the BO results
checkpoint = 100
decode_attempts = 5
max_n = 12
data_type = 'EQ'
hs, nz = args.hs, args.nz
optimizer = args.optimizer
seed_count = 30


class EvalLatent:
    def __init__(self, nz, eva):
        self.best_score = 0
        self.best_eq = None
        self.nz = nz
        self.eva = eva
        self.num_probes = 0

    def get_best_eq(self):
        return self.best_eq, self.best_score

    def get_probe_count(self):
        return self.num_probes

    def evaluate_latent_point(self, x_in=None, **latent_x):
        if x_in is None:
           latent_x_l = []
           for i in range(self.nz):
               latent_x_l.append(latent_x['x' + str(i).zfill(2)])
           latent_x_arr = np.array(latent_x_l).reshape((1, -1))
        else:
           latent_x_arr = np.array(x_in).reshape((1, -1))
        if is_cond:
            y_cond_single = self.eva.get_dcond()
            y_cond = torch.cuda.FloatTensor(
                np.broadcast_to(y_cond_single, shape=(latent_x_arr.shape[0], y_cond_single.shape[0])))
            valid_arcs = decode_from_latent_space(torch.FloatTensor(latent_x_arr).cuda(), model,
                                                  decode_attempts, max_n, False, data_type, y_cond=y_cond)
        else:
            valid_arcs = decode_from_latent_space(torch.FloatTensor(latent_x_arr).cuda(), model,
                                                  decode_attempts, max_n, False, data_type)

        cur_scores = []
        for i in range(len(valid_arcs)):
            arc = valid_arcs[i]
            # print(arc)
            if arc is not None:
                cur_score = self.eva.eval(arc)
            else:
                cur_score = 0
            cur_scores.append(cur_score)

        max_idx = np.array(cur_scores).argmax()
        max_score = np.array(cur_scores).max()
        best_arc = valid_arcs[max_idx]
        if max_score > self.best_score:
            self.best_score = max_score
            self.best_eq = best_arc
            print(self.best_score, self.best_eq)
        if max_score > 0.99:
            print('best eq ', best_arc)
            print(max_score)
        self.num_probes += 1
        return max_score


def dual_anneal(x_train, nz, eva, max_fun=1000, random_state=42):
    eval_latent = EvalLatent(nz, eva)

    x_mean = x_train.mean(0)
    x_std = x_train.std(0)
    x_upper = x_mean + 2*x_std
    x_lower = x_mean - 2*x_std
    pbounds = []
    for i in range(len(x_upper)):
        pbounds.append((x_lower[i], x_upper[i]))
    dual_annealing(lambda x: -eval_latent.evaluate_latent_point(x_in=x), bounds=pbounds,
                   maxfun=max_fun, initial_temp=5e4, seed=random_state, no_local_search=True)
    return eval_latent.get_best_eq()


def bayesian_opt(x_train, nz, eva, init_points=100, iterations=200, random_state=42):
    eval_latent = EvalLatent(nz, eva)

    # Bounded region of parameter space
    x_mean = x_train.mean(0)
    x_std = x_train.std(0)
    x_upper = x_mean + 3*x_std
    x_lower = x_mean - 3*x_std
    pbounds = {}
    for i in range(len(x_upper)):
        pbounds['x' + str(i).zfill(2)] = (x_lower[i], x_upper[i])
    print(pbounds)
    optimizer = BayesianOptimization(
        f=eval_latent.evaluate_latent_point,
        pbounds=pbounds,
        random_state=random_state
    )

    optimizer.maximize(init_points=init_points, n_iter=iterations)

    return eval_latent.get_best_eq()

# # do BO experiments with 10 random seeds
# for rand_idx in range(1,bo_seed_count + 1):
#
#
#     save_dir = 'bayesian_optimization/{}results_{}_{}/'.format(res_dir, save_appendix, rand_idx)  # where to save the BO results
#     if data_type == 'BN':
#         eva = Eval_BN(save_dir)  # build the BN evaluator
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # backup files
#     copy('bayesian_optimization/bo.py', save_dir)
#     if args.predictor:
#         copy('bayesian_optimization/run_pred_{}.sh'.format(data_type), save_dir)
#     elif args.vis_2d:
#         pass
#         #copy('bayesian_optimization/run_vis_{}.sh'.format(data_type), save_dir)
#     else:
#         copy('bayesian_optimization/run_bo_{}.sh'.format(data_type), save_dir)
#
#     # set seed
#     random_seed = rand_idx
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     np.random.seed(random_seed)


if __name__ == '__main__':
    if is_cond:
        eva = Eval_EQ(sr_dataset_path='sr_evaluation/sr_dataset_{}.csv'.format(dataset_num), embed_mode='nesymres',
                      res_path=data_dir)
    else:
        eva = Eval_EQ(sr_dataset_path='sr_evaluation/sr_dataset_{}.csv'.format(dataset_num), embed_mode=None)
    print('SR evaluation on dataset {}'.format(dataset_num))

    # load the decoder
    model = eval('DVAE')(
            max_n=max_n,
            nvt=9,
            START_TYPE=0,
            END_TYPE=1,
            hs=hs,
            nz=nz,
            bidirectional=True,
            cs=cond_size,
            cs_red=20
            )

    model.cuda()
    load_module_state(model, data_dir + 'model_checkpoint{}.pth'.format(checkpoint))

    data = loadmat(data_dir + '{}_latent_epoch{}.mat'.format(data_name, checkpoint))  # load train/test data
    X_train = data['Z_train']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(save_dir + 'best_arc_scores.txt'):
        os.remove(save_dir + 'best_arc_scores.txt')

    if optimizer == 'bayes':
        for seed in range(seed_count):
            best_arc, best_score = bayesian_opt(X_train, nz, eva, init_points=10, iterations=10, random_state=seed)
            with open(save_dir + 'best_arc_scores.txt', 'a') as score_file:
                score_file.write('{} , {:.4f}\n'.format(best_arc, best_score))
    elif optimizer == 'anneal':
        for seed in range(seed_count):
            best_arc, best_score = dual_anneal(X_train, nz, eva, max_fun=1000, random_state=seed)
            with open(save_dir + 'best_arc_scores.txt', 'a') as score_file:
                score_file.write('{} , {:.4f}\n'.format(best_arc, best_score))

