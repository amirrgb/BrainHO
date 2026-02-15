import os
import sys
import time
from datetime import datetime
import argparse
import logging
import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

import utils
import trainer
from dataset import (
    load_data_mdd_bd,
    load_data_abide_cc200,
    load_data_abide_aal116,
    load_data_rest_meta_mdd
)
from model import BrainHO, BrainHO_woHA, TF
from model_vis import BrainHO_vis
from vis import vis_epoch, vis_learned_composition_yeo7_style, export_subgraph_composition_to_csv, export_all_subgraphs_brainnet_clipped_aal

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Model training...")

# main
parser.add_argument('--session', default="training", type=str, help='session')
parser.add_argument('--seed', default=0, type=int, help='fix random seed')
parser.add_argument("--rootdir", default=".", type=str, help="project directory")
parser.add_argument("--logdir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument('--num_workers', default=0, type=int, help='for loading data')
parser.add_argument("--ckpt_path",
                    default="/home/path/path/ckptfile",
                    type=str, help="ckpt_path")
parser.add_argument("--cuda", default=0, type=int, help='cuda device')
parser.add_argument("--save_ckpt", action="store_true", help="use save ckpt")
parser.add_argument("--vis", action="store_true", help="use visualization")
parser.add_argument("--n_trials", default=100, type=int, help="number of trials for bayesian search")

# dataset
parser.add_argument("--dataset", default="abide_cc200", type=str, help="dataset name[mdd, bd, abide]")
parser.add_argument("--data_aug", action="store_true", help="use data_aug")
parser.add_argument("--dataset_path", default='/home/path/path/dataset/abide_cc200/abide.npy', type=str,
                    help="dataset abs_path\
                    MDD: /home/path/path/dataset/njmu_mdd/MDD_HC.npy \
                    BD: /home/path/path/dataset/njmu_bd/BD_HC.npy \
                    ABIDE: /home/path/path/dataset/abide_cc200/abide.npy  ")
parser.add_argument("--label_path", default='/home/path/path/dataset/njmu_mdd/MDD_HC_label.npy', type=str,
                    help="label abs_path\
                    MDD: /home/path/path/dataset/njmu_mdd/MDD_HC_label.npy \
                    BD: /home/path/path/dataset/njmu_bd/BD_HC_label.npy \
                    ")
parser.add_argument("--n_splits", default=5, type=int, help="k fold")
parser.add_argument("--cur_fold", default=0, type=int, help="cur_fold")
parser.add_argument("--mixup_alpha", default=0.5, type=float, help="mixup param")

# model
parser.add_argument("--model", default="BrainHO", type=str, help="model name")
parser.add_argument('--input_dim', default=200, type=int, help='input_dim')
parser.add_argument('--embed_dim', default=384, type=int, help='embed_dim')
parser.add_argument('--num_classes', default=2, type=int, help='num_classes')
parser.add_argument('--num_heads', default=8, type=int, help='num_heads')
parser.add_argument('--depth', default=2, type=int, help='model depth')
parser.add_argument("--dropout", default=0.0, type=float, help='dropout')
parser.add_argument("--attn_drop", default=0.0, type=float, help='attn_drop')
parser.add_argument("--drop_path", default=0.0, type=float, help='drop_path')
parser.add_argument('--num_node', default=200, type=int, help='num_node')
parser.add_argument('--num_sg', default=8, type=int, help='num_node')

# loss
parser.add_argument("--w_loss1", default=1.0, type=float, help="w_loss1")
parser.add_argument("--w_loss2", default=0.001, type=float, help="w_loss2")
parser.add_argument("--temperature", default=2.0, type=float, help="temperature")
parser.add_argument("--target_weight", default=0.2, type=float, help="temperature")
parser.add_argument("--center_step_div", default=4, type=int, help="temperature")
parser.add_argument("--slope", default=0.001, type=float, help="temperature")

# train
parser.add_argument("--batch_size", default=64, type=int, help="number of batch size")
parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
parser.add_argument("--val_epoch_freq", default=1, type=int, help="validation frequency")

# optim
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="optimization weight_decay")
parser.add_argument("--sche_name", default="cos", type=str, help="scheduler algorithm")
parser.add_argument("--sch_min_lr", default=1e-5, type=float, help="scheduler min learning rate")


def main_worker(args, logger):
    # logger.info('Starting train and test session !')
    utils.fix_random_seed(args.seed)
    # logger.info("current random seed: {}".format(args.seed))
    writer=None
    writer_path = os.path.join(args.logdir, 'session')
    writer = SummaryWriter(writer_path)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        torch.cuda.set_device(args.cuda)
        logger.info('The GPU device {} is set !'.format(gpu_name))
    # ---------------------------------------------------------------------------dataset
    if args.dataset == 'mdd' or args.dataset == 'bd':
        data_pearson, labels, ori_labels = load_data_mdd_bd(args.dataset_path, args.label_path)
    elif args.dataset == 'abide_cc200':
        data_pearson, labels, ori_labels, ori_labels_np, site_np, site_one_hot = load_data_abide_cc200(
            args.dataset_path)
    elif args.dataset == 'abide_cc200_nyu':
        data_pearson, labels, ori_labels, ori_labels_np, site_np, site_one_hot = load_data_abide_cc200(
            args.dataset_path, pick_site='nyu')
    elif args.dataset == 'abide_cc200_nyu_um_ucla':
        data_pearson, labels, ori_labels, ori_labels_np, site_np, site_one_hot = load_data_abide_cc200(
            args.dataset_path, pick_site='nyu_um_ucla')
    elif args.dataset == 'abide_aal116':
        data_pearson, labels, ori_labels = load_data_abide_aal116(args.dataset_path)
    elif args.dataset == 'abide_aal116_nyu':
        data_pearson, labels, ori_labels = load_data_abide_aal116(args.dataset_path, pick_site='NYU')
    elif args.dataset == 'rest-meta-mdd':
        data_pearson, labels, ori_labels_np, site_np = load_data_rest_meta_mdd(args.dataset_path)

    '''----------------------------------------------create dataset----------------------------------------------'''
    dataset = TensorDataset(data_pearson, labels)
    stratify_target = None
    if args.dataset == 'mdd' or args.dataset == 'bd':
        # 使用普通 KFold，不分层
        skf = KFold(n_splits=args.n_splits, shuffle=True)
        stratify_target = None
    else:
        # 其他数据集使用分层 KFold，以 site_np 为分层依据
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True)
        stratify_target = site_np
    # 所有测试结果
    all_fold_auc = []
    all_fold_acc = []
    all_fold_sen = []
    all_fold_spe = []
    all_fold_val_acc = []
    if args.vis:
        args.batch_size = 1
        total_main_attn_gsg_accum = None
        total_main_attn_sgn_accum = None
        total_main_attn_node_accum = None
        total_main_attn_g2node_accum = None
        full_folds_counts = 0

    # 5折
    if args.dataset == 'mdd' or args.dataset == 'bd':
        split_generator = skf.split(data_pearson)
    else:
        split_generator = skf.split(data_pearson, stratify_target)
    for fold, (train_val_idx, test_idx) in enumerate(split_generator):
        utils.fix_random_seed(args.seed)  # for same model
        logger.info(
            f"\n--------------------------------------------Fold {fold + 1}/{args.n_splits}--------------------------------------------")
        args.cur_fold = fold + 1
        if args.dataset == 'mdd' or args.dataset == 'bd':
            inner_stratify = None
        else:
            inner_stratify = site_np[train_val_idx]

        val_size = 0.25
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=inner_stratify
        )
        # 创建采样器
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        # print(f'test_idx: {test_idx}')
        # a=[]
        # b=a[1]
        # # 站点数
        # train_site, train_counts = np.unique(site_np, return_counts=True)
        # logger.info('all-----------')
        # for i in range(len(train_site)):
        #     logger.info(f"\n{train_site[i]}, {train_counts[i]}")
        #
        # train_site, train_counts = np.unique(site_np[train_idx], return_counts=True)
        # logger.info('train-----------')
        # for i in range(len(train_site)):
        #     logger.info(f"\n{train_site[i]}, {train_counts[i]}")
        # val_site, val_counts = np.unique(site_np[val_idx], return_counts=True)
        # logger.info('val------------')
        # for i in range(len(val_site)):
        #     logger.info(f"{val_site[i]}, {val_counts[i]}")
        # test_site, test_counts = np.unique(site_np[test_idx], return_counts=True)
        # logger.info('test-----------')
        # for i in range(len(test_site)):
        #     logger.info(f"\n{test_site[i]}, {test_counts[i]}")
        # continue
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        # logger.info('The dataloader is set !')
        '''----------------------------------------------create model----------------------------------------------'''
        if args.model == 'BrainHO':
            model = BrainHO(input_dim=args.input_dim,
                                        embed_dim=args.embed_dim,
                                        num_classes=args.num_classes,
                                        num_heads=args.num_heads,
                                        depth=args.depth,
                                        dropout=args.dropout,
                                        attn_drop=args.attn_drop,
                                        drop_path=args.drop_path,
                                        num_node=args.num_node,
                                        num_sg=args.num_sg,
                                        )
        elif args.model == 'BrainHO_woHA':
            model = BrainHO_woHA(input_dim=args.input_dim,
                                        embed_dim=args.embed_dim,
                                        num_classes=args.num_classes,
                                        num_heads=args.num_heads,
                                        depth=args.depth,
                                        dropout=args.dropout,
                                        attn_drop=args.attn_drop,
                                        drop_path=args.drop_path,
                                        num_node=args.num_node,
                                        num_sg=args.num_sg,
                                        )
        elif args.model == 'TF':
            model = TF(input_dim=args.input_dim,
                                        embed_dim=args.embed_dim,
                                        num_classes=args.num_classes,
                                        num_heads=args.num_heads,
                                        depth=args.depth,
                                        dropout=args.dropout,
                                        attn_drop=args.attn_drop,
                                        drop_path=args.drop_path,
                                        num_node=args.num_node,
                                        num_sg=args.num_sg,
                                        )
        elif args.model == 'BrainHO_vis':
            model = BrainHO_vis(input_dim=args.input_dim,
                                        embed_dim=args.embed_dim,
                                        num_classes=args.num_classes,
                                        num_heads=args.num_heads,
                                        depth=args.depth,
                                        dropout=args.dropout,
                                        attn_drop=args.attn_drop,
                                        drop_path=args.drop_path,
                                        num_node=args.num_node,
                                        num_sg=args.num_sg,
                                        )
        model.cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info('The model is set !')
        # logger.info("Model total parameters count: {} M".format(pytorch_total_params/ 1_000_000.0))
        '''----------------------------------------------create optim----------------------------------------------'''
        # total step
        total_training_steps = args.max_epochs * len(train_loader)
        # optim
        optimizer = None
        scheduler = None
        if args.optim_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, betas=(0.9, 0.999),
                                          weight_decay=args.weight_decay)
        elif args.optim_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.weight_decay)
        if args.sche_name == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps,
                                          eta_min=args.sch_min_lr)  # 100epoch，min_lr 1e-6
        # logger.info('The optimizer is set !')
        start_epoch = 0
        '''----------------------------------------------visulization----------------------------------------------'''
        if args.vis:
            # load weight
            best_ckpt_pth = args.ckpt_path + str(args.cur_fold) + '.pt'
            ckpt_path_fold = os.path.join('', best_ckpt_pth)
            if not os.path.exists(ckpt_path_fold):
                logger.warning(f"no ckpt:{ckpt_path_fold}, skip Fold")
                continue
            model_dict = torch.load(ckpt_path_fold, map_location=torch.device('cpu'))
            state_dict = model_dict["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            logger.info("loaded best model weights !!!")
            model.cuda()
            # forzen model
            for param in model.parameters():
                param.requires_grad = False
            auc, acc, sen, spec, total_fold_attn_gsg_accum, total_fold_attn_sgn_accum, total_fold_attn_node_accum, total_fold_attn_g2node_accum, total_samples_count = vis_epoch(model, test_loader, args=args, tvt='test')

            if total_main_attn_gsg_accum is None:
                total_main_attn_gsg_accum = total_fold_attn_gsg_accum
            else:
                total_main_attn_gsg_accum += total_fold_attn_gsg_accum

            if total_main_attn_sgn_accum is None:
                total_main_attn_sgn_accum = total_fold_attn_sgn_accum
            else:
                total_main_attn_sgn_accum += total_fold_attn_sgn_accum

            if total_main_attn_node_accum is None:
                total_main_attn_node_accum = total_fold_attn_node_accum
            else:
                total_main_attn_node_accum += total_fold_attn_node_accum

            if total_main_attn_g2node_accum is None:
                total_main_attn_g2node_accum = total_fold_attn_g2node_accum
            else:
                total_main_attn_g2node_accum += total_fold_attn_g2node_accum

            full_folds_counts += total_samples_count

            logger.info(f"fetch Fold {fold + 1} ASD vis data")
            logger.info(f"vis_acc = {acc * 100:.3f}")
            continue # 继续提取下一折
        else:
            '''----------------------------------------------training----------------------------------------------'''
            best_model_weights, best_val_acc, best_val_epoch = trainer.run_train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                start_epoch=start_epoch,
                total_training_steps=total_training_steps,
                logger=logger,
                writer=writer
            )
            model.load_state_dict(best_model_weights)
            test_auc, test_acc, test_sen, test_spe = trainer.test_epoch(model, test_loader)
            logger.info("Val Stage,  Epoch: {}".format(best_val_epoch))
            logger.info("Val Stage,  ACC: {:.3f}".format(best_val_acc * 100))
            logger.info('************************** Test  **************************')
            logger.info("Test Stage, ACC: {:.3f}, AUC: {:.3f}, SEN: {:.3f}, SPE: {:.3f}".format(
                test_acc * 100, test_auc * 100, test_sen * 100, test_spe * 100))
            logger.info('------------------------------------------------------------')
            all_fold_acc.append(test_acc)
            all_fold_auc.append(test_auc)
            all_fold_sen.append(test_sen)
            all_fold_spe.append(test_spe)
            all_fold_val_acc.append(best_val_acc)

    if args.vis:
        logger.info(f"所有 Fold 处理完毕，正在计算平均值并绘图")
        avg_attn_gsg = total_main_attn_gsg_accum / full_folds_counts
        avg_attn_sgn = total_main_attn_sgn_accum / full_folds_counts
        avg_attn_node = total_main_attn_node_accum / full_folds_counts
        avg_attn_g2node = total_main_attn_g2node_accum / full_folds_counts
        print(f'avg_attn_gsg: {avg_attn_gsg.shape}')
        print(f'avg_attn_gsg: {avg_attn_gsg}')
        print(f'avg_attn_sgn: {avg_attn_sgn.shape}')
        print(f'avg_attn_node: {avg_attn_node.shape}')
        print(f'avg_attn_g2node: {avg_attn_g2node.shape}')
        '''----------------------------------------------key sg----------------------------------------------'''

        vis_learned_composition_yeo7_style(avg_attn_sgn)

        export_subgraph_composition_to_csv(
            avg_attn_sgn,
            mapping_source='/home/path/path/dataset/abide_cc200/node_clus_map.pickle',
            save_path='sg_composition_data_mdd.csv'
        )

        export_all_subgraphs_brainnet_clipped_aal(
            avg_attn_sgn,
            avg_attn_node,
            save_dir="BrainNet_AAL_Results-mdd",
            aal_mapping_file="/home/path/path/CC200-to-AAL116-Mapping/CC200_to_AAL116_Mapping_with_Names.csv"
        )

        return 0

    if args.vis == False:
        auc_all = np.array(all_fold_auc)
        acc_all = np.array(all_fold_acc)
        sen_all = np.array(all_fold_sen)
        spe_all = np.array(all_fold_spe)
        auc_mean = np.mean(auc_all)
        acc_mean = np.mean(acc_all)
        sen_mean = np.mean(sen_all)
        spe_mean = np.mean(spe_all)
        auc_std = np.std(auc_all)
        acc_std = np.std(acc_all)
        sen_std = np.std(sen_all)
        spe_std = np.std(spe_all)

        val_acc_all = np.array(all_fold_val_acc)
        val_acc_mean = np.mean(val_acc_all)
        val_acc_std = np.std(val_acc_all)

        logger.info('************************** Metric  **************************')
        logger.info(f"val ACC : mean = {val_acc_mean * 100:.3f}, std = {val_acc_std * 100:.3f}")
        logger.info('--------------------------------------------------------------')
        logger.info(f"ACC : mean = {acc_mean * 100:.3f}, std = {acc_std * 100:.3f}")
        logger.info(f"AUC : mean = {auc_mean * 100:.3f}, std = {auc_std * 100:.3f}")
        logger.info(f"SEN : mean = {sen_mean * 100:.3f}, std = {sen_std * 100:.3f}")
        logger.info(f"SPE : mean = {spe_mean * 100:.3f}, std = {spe_std * 100:.3f}")
        logger.info('--------------------------------------------------------------')
        if writer is not None:
            writer.close()
        return acc_mean * 100

    if writer is not None:
        writer.close()

    return 0


def main():
    args = parser.parse_args()
    utils.fix_random_seed(args.seed)

    # log
    timestamp = time.time()
    time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    time_str = time_str + '_' + args.session
    log_path = os.path.join(args.rootdir, args.logdir, time_str)
    args.logdir = log_path
    os.makedirs(log_path, exist_ok=True)
    logger = utils.init_log('global', log_path, logging.INFO, session=args.session)
    logger.propagate = 0
    '''----------------------------------------------log key params----------------------------------------------'''
    args_dict = vars(args)
    explicit_args = {}
    for k, v in args_dict.items():
        if f'--{k}' in sys.argv or f'--{k.replace("_", "-")}' in sys.argv:
            explicit_args[k] = v
    utils.log_args_table(explicit_args, logger)
    '''---------------------------------one session---------------------------------'''

    utils.log_args_table(vars(args), logger)
    main_worker(args=args, logger=logger)

    # total_time
    total_time = (time.time() - timestamp) / 60
    logger.info("total_time: {:.3f} min".format(total_time))
    logger.info("main Finished !")


if __name__ == "__main__":
    main()