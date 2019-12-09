import csv
import glob
import os
import pandas as pd
import numpy as np


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=',')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    if exclude_from is not None:
        sort_val = sort_val[sort_idxs]
        col = col[sort_val >= exclude_from]
    return col


def write_csv_row(resultCSV, res_dict, headers):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(res_dict)


headers_bin = ['folder_p', 'file_p', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'ROC AUC']
headers_reg = ['folder_p', 'file_p', 'Accuracy', 'RMSE', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'ROC AUC']

prefix = '\\\\?\\'
fps = [r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed_experiment2\summary\one_slice',
       r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed_experiment2\summary\dist_10',
       r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_10-90_lower_lr_imgnet_fixed_experiment2\summary\nine_slice',
       r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_10-90_lower_lr_imgnet_fixed_experiment2\summary\two_seven_slice']

fps = [r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_resnet152_experiment2\summary\one_slice',
       r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_resnet152_experiment2\summary\dist_10']

binary_files = ['binary_non_pretrained.csv', 'binary_pretrained.csv']
regression_files = ['non_pretrained_regression.csv', 'pretrained_regression.csv']

csv_out = r'C:\Users\Fabian\Desktop'
field_pairs_bin = [['mean_test_acc', 'std_test_acc'], ['mean_sensitivity', 'std_sensitivity'],
                   ['mean_specificity', 'std_specificity'], ['mean_ppv', 'std_ppv'], ['mean_npv', 'std_npv'],
                   ['mean_roc_auc', 'std_roc_auc']]


csv_out_f = os.path.join(csv_out, 'bin.csv')
for fp in fps:
    fp = prefix+fp
    for f in binary_files:
        res_dict = {}
        csv_f = os.path.join(fp, f)
        res_dict[headers_bin[0]] = os.path.basename(fp)
        res_dict[headers_bin[1]] = f
        for head, field_pair in zip(headers_bin[2:], field_pairs_bin):
            col1 = get_csv_column(csv_f, field_pair[0])
            col2 = get_csv_column(csv_f, field_pair[1])
            if head == 'ROC AUC' or head == 'RMSE':
                result_str = f'{col1[-1]:.4f} ({col2[-1]:.4f} SD)'
            else:
                result_str = f'{col1[-1] * 100:.2f}% ({col2[-1] * 100:.2f}% SD)'
            res_dict[head] = result_str
        write_csv_row(csv_out_f, res_dict, headers_bin)

field_pairs_reg = [['mean_test_label_acc_test', 'std_test_label_acc_test'], ['mean_rmsetest_acc', 'std_rmsetest_acc'],
                    ['mean_sensitivity', 'std_sensitivity'],
                   ['mean_specificity', 'std_specificity'], ['mean_ppv', 'std_ppv'], ['mean_npv', 'std_npv'],
                   ['mean_roc_auc', 'std_roc_auc']]


csv_out_f = os.path.join(csv_out, 'reg.csv')
for fp in fps:
    fp = prefix+fp
    for f in regression_files:
        res_dict = {}
        csv_f = os.path.join(fp, f)
        res_dict[headers_reg[0]] = os.path.basename(fp)
        res_dict[headers_reg[1]] = f
        for head, field_pair in zip(headers_reg[2:], field_pairs_reg):
            col1 = get_csv_column(csv_f, field_pair[0])
            col2 = get_csv_column(csv_f, field_pair[1])
            if head == 'ROC AUC' or head == 'RMSE':
                result_str = f'{col1[-1]:.4f} ({col2[-1]:.4f} SD)'
            else:
                result_str = f'{col1[-1]*100:.2f}% ({col2[-1]*100:.2f}% SD)'
            res_dict[head] = result_str
        write_csv_row(csv_out_f, res_dict, headers_reg)