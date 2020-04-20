import os, sys, inspect
# this is done to run things from console
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dl.neural_network.train_h5 import train_h5
from dl.neural_network.train_h5_multi import train_h5 as train_h5_multi
import GPUtil
import multiprocessing as mp
import time
import datetime
from glob import glob


def job_generator(jobs):
    for job in jobs:
        yield job


def run_jobs(jobs):
    device_ids = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
    print(device_ids)
    job_gen = job_generator(jobs)
    process_dict = {}
    while True:
        try:
            if process_dict == {}:
                for device in device_ids:
                    job = next(job_gen)
                    print(f"Running {job} on GPU {device}")
                    sub_proc = mp.Process(target=train_h5, args=[job[0]], kwargs={'gpu_device': device, **job[1]})
                    process_dict[str(device)] = sub_proc
                    sub_proc.start()
            for device, proc in process_dict.items():
                if not proc.is_alive():
                    job = next(job_gen)
                    print(f"Running {job} on GPU {device}")
                    sub_proc = mp.Process(target=train_h5, args=[job[0]], kwargs={'gpu_device': device, **job[1]})
                    process_dict[str(device)] = sub_proc
                    sub_proc.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in process_dict.values():
        proc.join()


if __name__ == '__main__':
    full_start = time.time()
    # get activations
    sub_folders = ['/scratch/reith/fl/experiments/feature_acs_multi']
    for sub in sub_folders:
        seed = 10
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_0_79_suvr', 'label_amyloid'], 'regression': True,
             'lr': 0.0001, 'seed': seed, 'save_model': True, 'batch_size': 32, 'extract_features': True, 'use_multi_resnet': False,
             'threshold': 0.79}]
        h5_files = glob(f'{sub}/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")


r'''
################LATER#################################
if __name__ == '__main__':
    full_start = time.time()
    # get activations
    sub_folders = ['/scratch/reith/fl/experiments/feature_acs_multi']
    for sub in sub_folders:
        seed = 10
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['multi_res_suvr_age_apoe', 'label_amyloid'], 'regression': True,
             'lr': 0.0001, 'seed': seed, 'save_model': True, 'batch_size': 32, 'extract_features': True, 'use_multi_resnet': True,
             'threshold': 0.79}]
        h5_files = glob(f'{sub}/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
###################################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run on smaller resnext with higher batch size
    super_folder = '/scratch/reith/fl/experiments/seeds_resnext'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    sub_folders.sort(key=lambda x: int(x.split('_')[-1]))
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
##############################################################3
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_10-90_resnext'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1}]
            # {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1},
            # {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1},
            # {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 1}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
'''

r'''
###############################################
######Past runs################################
###############################################
if __name__ == '__main__':
    full_start = time.time()
    # get activations
    sub_folders = ['/scratch/reith/fl/experiments/feature_acs_multi']
    for sub in sub_folders:
        seed = 10
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_0_79_suvr', 'label_amyloid'], 'regression': True,
             'lr': 0.0001, 'seed': seed, 'save_model': True, 'batch_size': 32, 'extract_features': True,
             'threshold': 0.79}]
        h5_files = glob(f'{sub}/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
##############################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run on smaller resnext with higher batch size
    super_folder = '/scratch/reith/fl/experiments/seeds_resnet152'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    sub_folders.sort(key=lambda x: int(x.split('_')[-1]))
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnet152': True, 'batch_size': 32},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnet152': True, 'batch_size': 32},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnet152': True, 'batch_size': 32},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnet152': True, 'batch_size': 32}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
$$$$$$$$$$$$$$$Transfer experiment $$$$$$$$$$$
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_10-90_lower_lr_transfer'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")


if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_lower_lr_transfer'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
####################################################################3
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_10-90_resnext'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 32},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 32},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 32},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnext': True, 'batch_size': 32}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
#########################################################3
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_10-90_lower_lr_imgnet_fixed'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")


if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_lower_lr_imgnet_fixed'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_10-90_lower_lr_imgnet_fixed'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
##########################################################
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_10-90_lower_lr_imgnet_fixed'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")


if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_lower_lr_imgnet_fixed'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
######################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_lower_lr'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds_more_dists'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_10']
    # sub_folders = ['/scratch/reith/fl/experiments/transfer_experiment/seed_higher_bs_freeze_5_epochs_10']
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001, 'seed': seed, 'freeze_epochs': 5, 'batch_size': 64},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.001, 'seed': seed, 'freeze_epochs': 5, 'batch_size': 64}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    super_folder = '/scratch/reith/fl/experiments/seeds2'
    sub_folders = glob(os.path.join(super_folder, '*seed*'))
    suf_folders = '/scratch/reith/fl/experiments/transfer_experiment/seed_10'
    print(sub_folders)
    for sub in sub_folders:
        seed = int(sub.split('_')[-1])
        jobs = [
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001, 'seed': seed},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.001, 'seed': seed},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_amyloid'], 'lr': 0.001, 'seed': seed}]
        h5_files = glob(f'{sub}/*_*/*.h5')
        # import pdb; pdb.set_trace()
        print(h5_files)
        for h5_file in h5_files:
            process_jobs = [(h5_file, job) for job in jobs]
            print(process_jobs)
            run_jobs(process_jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
############################################################33
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_site_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_site_three_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/dist_10_incl_subjects_site_three_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/dist_20_incl_subjects_site_three_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/dist_40_incl_subjects_site_three_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_site_three_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_site_three_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001}]
###########################################################################################3
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
#############################################################################3
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True,
             'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True,
             'lr': 0.00001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True,
             'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True,
             'lr': 0.0001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3,
             'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3,
             'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/incl_subjects_one_slices_dataset_full/slice_data_subj.h5'
    jobs = [{'extra_info': '', 'pretrained': True},
            {'extra_info': '', 'pretrained': False},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3,
             'rate_of_decrease': 0.33},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3,
             'rate_of_decrease': 0.33}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

'''




r'''
###############################################
##############Past runs bad subject separation:
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
#############################################################################3
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001}]
    

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': 'pretrained_20bins', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'non_pretrained_20bins', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'pretrained_50epochs_20bins', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'non_pretrained_50epochs_20bins', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'pretrained_0_33_decrease_20bins', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'non_pretrained_0_33_decrease_20bins', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': 'pretrained', 'pretrained': True},
            {'extra_info': 'non_pretrained', 'pretrained': False},
            {'extra_info': 'pretrained_50epochs', 'pretrained': True, 'num_epochs': 50},
            {'extra_info': 'non_pretrained_50epochs', 'pretrained': False, 'num_epochs': 50},
            {'extra_info': 'pretrained_0_33_decrease', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33},
            {'extra_info': 'non_pretrained_0_33_decrease', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
'''