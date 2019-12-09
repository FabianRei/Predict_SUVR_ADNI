import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
import re
from fnmatch import fnmatch
from fnmatch import filter
import csv


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


def viz_training(folder, identifier='', sort_by='epoch', title='default', train_acc='train_acc', test_acc='test_acc',
                 fname_addition=""):
    csv_path = glob(os.path.join(folder, f"*{identifier}*.csv"))[0]
    fname = f"viz_of_{identifier}{fname_addition}.png"
    fig = plt.figure()
    plt.xlabel('Epochs')
    if fnmatch(identifier, '*reg*') and fname_addition == "":
        plt.ylabel("Loss (MSE)")
    else:
        plt.ylabel('Accuracy')
    # num = folder_path.split('_')[-1]
    if title == 'default':
        title = f"Training progression for {identifier}{fname_addition}"
    plt.title(title)
    plt.grid(which='both')
    train_acc = get_csv_column(csv_path, train_acc, sort_by=sort_by)
    test_acc = get_csv_column(csv_path, test_acc, sort_by=sort_by)
    epochs = get_csv_column(csv_path, 'epoch', sort_by=sort_by)
    epochs += 1
    plt.plot(epochs, train_acc, label='Train data')
    plt.plot(epochs, test_acc, label='Test data')

    plt.legend(frameon=True, loc='upper left', fontsize='small')
    fig.savefig(os.path.join(folder, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


def viz_training_from_csv(csv_path, train_acc, test_acc, include_loss, train_loss='', test_loss=''):
    csv_name = os.path.basename(csv_path).split('.')[0]
    sort_by='epochs'
    fname = f"viz_of_{csv_name}_loss_{include_loss}.png"
    out_name = os.path.join(os.path.dirname(csv_path), fname)
    fig, ax1 = plt.subplots()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ax1.yaxis.tick_right()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0.5, 1.05])
    title = f"Training progression for {csv_name}"
    # plt.title(title)
    plt.grid(which='both')
    train_acc = get_csv_column(csv_path, train_acc, sort_by=sort_by)
    test_acc = get_csv_column(csv_path, test_acc, sort_by=sort_by)
    epochs = get_csv_column(csv_path, 'epochs', sort_by=sort_by)
    ax1.plot(epochs, train_acc, label='Train data accuracy', color=colors[0])
    ax1.plot(epochs, test_acc, label='Test data accuracy', color=colors[1])
    # loss = get_csv_column(csv_path, train_loss, sort_by=sort_by)
    # ax1.plot(epochs, loss, label='Test data loss', color=colors[3])
    if include_loss:
        ax2 = ax1.twinx()
        if fnmatch(csv_name, '*reg*'):
            ax2.set_ylabel('Loss (MSE)')
            ax2.set_ylim([-0.005, 0.4])
        else:
            ax2.set_ylabel('Loss (NLL)')
            ax2.set_ylim([-0.05, 0.6])
        loss = get_csv_column(csv_path, train_loss, sort_by=sort_by)
        ax2.plot(epochs, loss, label='Train data loss', color=colors[2])
        if test_loss != '':
            test_loss = get_csv_column(csv_path, test_loss, sort_by=sort_by)
            ax2.plot(epochs, test_loss, label='Test data loss', color=colors[3])
    # plt.legend(frameon=True, loc='upper left', fontsize='small')
    # plt.tight_layout()
    fig.savefig(out_name, dpi=200)
    # fig.show()
    print('done!')


def viz_training_data(out_folder, train_acc, test_acc, epochs, identifier='', title='default', fname_addition=''):
    fname = f"viz_of_{identifier}{fname_addition}.png"
    fig = plt.figure()
    plt.xlabel('Epochs')
    if fnmatch(identifier, '*reg*') and fname_addition == "":
        plt.ylabel("Loss (MSE)")
    else:
        plt.ylabel('Accuracy')
    # num = folder_path.split('_')[-1]
    if title == 'default':
        title = f"Training progression for {identifier}{fname_addition}"
    plt.title(title)
    plt.grid(which='both')
    plt.plot(epochs, train_acc, label='Train data')
    plt.plot(epochs, test_acc, label='Test data')
    plt.legend(frameon=True, loc='upper left', fontsize='small')
    fig.savefig(os.path.join(out_folder, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


def find_all_identifiers(folder, file_ending='.csv', within_file_pattern=''):
    identifiers = []
    files = [os.path.basename(f) for f in glob(os.path.join(folder, f'*{file_ending}'))]
    for file in files:
        ids = re.findall(rf'\d+-\d+_\d+-\d+.*{within_file_pattern}.*.csv', file)
        ids = [id[:-4] for id in ids]
        identifiers.extend(ids)
    return identifiers


def write_csv_row(resultCSV, testAcc, accOptimal, d1, d2, dataContrast, nn_dprime):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index',
                   'contrast', 'nn_dprime']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1,
                         'optimal_observer_d_index': d2, 'contrast': dataContrast,
                         'nn_dprime': nn_dprime})


if __name__ == '__main__':
    super_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr\summary'
    fpaths = glob(os.path.join(super_path, '*_*'))
    for fpath in fpaths:
        # binary case
        csvs = glob(os.path.join(fpath, '*binary_*pretrained.csv'))
        for c in csvs:
            viz_training_from_csv(c, 'mean_train_acc', 'mean_test_acc', include_loss=True, train_loss='mean_train_loss')
        csvs = glob(os.path.join(fpath, '*pretrained_regression.csv'))
        for c in csvs:
            viz_training_from_csv(c, 'mean_test_label_acc_train', 'mean_test_label_acc_test', include_loss=True, train_loss='mean_train_acc', test_loss='mean_test_acc')
    print("nice")









r'''
##############PAST RUNS###################################

    for f in fpaths:
        folder = glob(os.path.join(f, '*_*'))
        for p in folder:
            fpath = p
            ids = find_all_identifiers(fpath)
            for ident in ids:
                viz_training(fpath, ident)
            ids_bin = find_all_identifiers(fpath, file_ending='.csv', within_file_pattern='bin')
            ids_reg = find_all_identifiers(fpath, file_ending='.csv', within_file_pattern='reg')
            for ident in ids_reg:
                viz_training(fpath, ident, train_acc='test_label_acc_train', test_acc='test_label_acc_test', fname_addition='_reg_test')
            for ident in ids_bin:
                viz_training(fpath, ident, train_acc='test_label_acc_train', test_acc='test_label_acc_test',
                             fname_addition='_bin_test')
#####################################################
if __name__ == '__main__':
    super_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds'
    fpaths = glob(os.path.join(super_path, '*seed*'))
    
    fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments'
    paths = glob(fpath + r"\*site*")
    for p in paths:

        # fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\more_one_slice_dataset'
        # fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\incl_subjects_one_slices_dataset_full'
        # fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\dist_40_incl_subjects_site_three_slices_dataset_full'
        # identifier = r'07-30_09-40_pretrain_normalizeData'
        # viz_training(fpath, identifier)
        fpath = p
        ids = find_all_identifiers(fpath)
        for ident in ids:
            viz_training(fpath, ident)
        ids_bin = find_all_identifiers(fpath, file_ending='.csv', within_file_pattern='bin')
        ids_reg = find_all_identifiers(fpath, file_ending='.csv', within_file_pattern='reg')
        for ident in ids_reg:
            viz_training(fpath, ident, train_acc='test_label_acc_train', test_acc='test_label_acc_test', fname_addition='_reg_test')
        for ident in ids_bin:
            viz_training(fpath, ident, train_acc='test_label_acc_train', test_acc='test_label_acc_test',
                         fname_addition='_bin_test')
'''