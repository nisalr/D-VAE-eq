import sys
import os

import glob
import pandas as pd
from calc_solution_rate import solution_match

results_path = sys.argv[1]
results_subfolder = results_path.split('/')[-1]
out_path = 'bo_agg_results/'
print(results_path)

results_folders = glob.glob(results_path + '_*')
print(results_folders)

all_results_l = []
for folder in results_folders:
    best_score_files = glob.glob(folder + '/*/best_arc_scores.txt')
    df_list = []
    if len(best_score_files) > 0:
        for file_path in best_score_files:
            df_cur = pd.read_csv(file_path, header=None)
            file_name = file_path.split('/')[1]
            file_no = int(file_name.split('_')[-1])
            true_eq = pd.read_csv('../sr_evaluation/sr_dataset_{}.csv'.format(file_no), nrows=1)['eq'].iloc[0]
            df_cur['file_no'] = file_no
            df_cur['file_name'] = file_name
            df_cur.columns = ['pred_eq', 'score', 'file_no', 'file_name']
            df_cur['true_eq'] = true_eq
            df_list.append(df_cur)
        results_df = pd.concat(df_list, axis=0).sort_values('score', ascending=False).iloc[[0]]
        all_results_l.append(results_df)
all_results = pd.concat(all_results_l).sort_values('file_no')
all_results['is_solution'] = all_results.apply(
    lambda x: solution_match(x['true_eq'], x['pred_eq']), axis=1)
print(all_results)
all_results.to_csv(out_path + results_subfolder + '.csv', index=False)

# for file_path in best_score_files:
#     df_cur = pd.read_csv(file_path, header=None)
#     file_name = file_path.split('/')[1]
#     file_no = int(file_name.split('_')[-1])
#     true_eq = pd.read_csv('../sr_evaluation/sr_dataset_{}.csv'.format(file_no), nrows=1)['eq'].iloc[0]
#     df_cur['file_no'] = file_no
#     df_cur['file_name'] = file_name
#     df_cur.columns = ['pred_eq', 'score', 'file_no', 'file_name']
#     df_cur['true_eq'] = true_eq
#     df_list.append(df_cur)
#
# df = pd.concat(df_list).reset_index(drop=True)
# idx = df.groupby(['file_name'])['score'].transform(max) == df['score']
# df_out = df[idx]
# df_out = df_out.drop_duplicates()
# # df_out = df_out.groupby(['file_no', 'file_name', 'pred_eq', 'true_eq', 'score']).first()
# print(df_out)
# # df_out = df.groupby(['file_name', 'true_eq', 'file_no']).max()#.sort_values('file_no')
# # print(df_out)
# df_out.to_csv(out_path + results_subfolder + '.csv', index=False)




