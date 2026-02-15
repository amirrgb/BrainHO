import time
import copy
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from utils import AverageMeter, SigmoidScheduler, save_checkpoint
from dataset import continus_mixup_data
from losses import compute_cross_entropy_Orthogonal_Regularization_loss, compute_distillation_loss


def train_epoch(model, loader, optimizer, scheduler, w_loss2_scheduler, args):
    model.train()
    loss_avg = AverageMeter()
    loss_cls_avg = AverageMeter()
    loss_aux_avg = AverageMeter()
    loss_orth_avg = AverageMeter()
    loss_kd_avg = AverageMeter()

    loss_ce = torch.nn.CrossEntropyLoss(reduction='mean')

    for idx, batch_data in enumerate(loader):

        data_pearson, labels = batch_data
        data_pearson, labels = data_pearson.cuda(), labels.cuda()

        # print(f'data_p_BrainHO: {data_pearson[0, 5, 6]}')
        # a = []
        # b = a[1]

        if args.data_aug:
            data_pearson, labels = continus_mixup_data(data_pearson, alpha=args.mixup_alpha, y=labels)

        pred, pred_node_aux, x_sg = model(data_pearson)

        # loss
        loss_cls = loss_ce(pred, labels)
        loss_orth = compute_cross_entropy_Orthogonal_Regularization_loss(x_sg)
        loss_aux = loss_ce(pred_node_aux, labels)  # aux Loss
        loss_kd = compute_distillation_loss(pred_node_aux, pred, temp=args.temperature)
        pred_dtc = pred.clone().detach()


        # warmup w_loss2
        args.w_loss2 = w_loss2_scheduler.step()
        loss = loss_cls + args.w_loss1 * loss_orth + args.w_loss2 * loss_kd + loss_aux

        loss_avg.update(loss.item(), n=args.batch_size)
        loss_cls_avg.update(loss_cls.item(), n=args.batch_size)
        loss_aux_avg.update(loss_aux.item(), n=args.batch_size)
        loss_orth_avg.update(loss_orth.item(), n=args.batch_size)
        loss_kd_avg.update(loss_kd.item(), n=args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss_avg.avg, loss_cls_avg.avg, loss_aux_avg.avg, loss_orth_avg.avg, loss_kd_avg.avg



def test_epoch(model, loader):
    model.eval()

    # 用于存储一个 epoch 内所有的预测概率
    all_pred_probs = []
    # 用于存储一个 epoch 内所有的真实标签
    all_true_labels = []

    with torch.no_grad():

        for idx, batch_data in enumerate(loader):

            data_pearson, labels = batch_data
            data_pearson, labels = data_pearson.cuda(), labels.cuda()

            pred, _, _ = model(data_pearson)

            # Metric
            pred_probs_all_classes = torch.softmax(pred, dim=1)
            y_scores_batch = pred_probs_all_classes[:, 1]
            y_true_batch = labels[:, 1]
            all_pred_probs.append(y_scores_batch.cpu().detach().numpy())
            all_true_labels.append(y_true_batch.cpu().numpy())


    # AUC-ROC ACC SEN SPEC
    epoch_pred_probs = np.concatenate(all_pred_probs)
    epoch_true_labels = np.concatenate(all_true_labels).astype(int)
    auc = roc_auc_score(epoch_true_labels, epoch_pred_probs)
    threshold = 0.5
    epoch_pred_labels = (epoch_pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(epoch_true_labels, epoch_pred_labels).ravel()
    total_samples = tp + tn + fp + fn
    acc = (tp + tn) / total_samples
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return auc, acc, sen, spec


def run_train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        args,
        start_epoch,
        total_training_steps,
        logger,
        writer=None
):
    best_val_acc = 0.0
    best_val_epoch = 0.0
    best_model = None

    w_loss2_scheduler = SigmoidScheduler(target_weight=args.target_weight, center_step=total_training_steps / args.center_step_div, slope=args.slope)

    for epoch in range(start_epoch, args.max_epochs):
        epoch_time = time.time()

        loss_avg, loss_cls_avg, loss_aux_avg, loss_orth_avg, loss_kd_avg = train_epoch(
            model, train_loader, optimizer=optimizer, scheduler=scheduler, w_loss2_scheduler=w_loss2_scheduler, args=args)

        logger.info(
            "Train Epoch: {}/{}, loss: {:.4f}, loss_cls: {:.4f}, loss_aux: {:.4f}, loss_orth: {:.4f}, loss_kd: {:.4f}, time: {:.3f} s".format(
                epoch + 1, args.max_epochs, loss_avg, loss_cls_avg, loss_aux_avg, loss_orth_avg, loss_kd_avg, time.time() - epoch_time))


        writer.add_scalar('fold_'+str(args.cur_fold)+'_'+'Train_stage/loss', loss_avg, epoch + 1)
        writer.add_scalar('fold_'+str(args.cur_fold)+'_'+'Train_stage/loss_cls', loss_cls_avg, epoch + 1)
        writer.add_scalar('fold_'+str(args.cur_fold)+'_'+'Train_stage/loss_aux', loss_aux_avg, epoch + 1)
        writer.add_scalar('fold_'+str(args.cur_fold)+'_'+'Train_stage/loss_orth', loss_orth_avg, epoch + 1)
        writer.add_scalar('fold_'+str(args.cur_fold)+'_'+'Train_stage/loss_kd', loss_kd_avg, epoch + 1)

        if (epoch + 1) % args.val_epoch_freq == 0:
            val_auc, val_acc, val_sen, val_spe = test_epoch(model, val_loader)
            # just for watching performance gap between val and test (BNT stratified sampling)
            test_auc, test_acc, test_sen, test_spe = test_epoch(model, test_loader)
            logger.info("Val Epoch: {}, ACC: {:.3f}, AUC: {:.3f}, SEN: {:.3f}, SPE: {:.3f}".format(
                epoch + 1, val_acc * 100, val_auc * 100, val_sen * 100, val_spe * 100))
            writer.add_scalar('fold_' + str(args.cur_fold) + '_' + 'Val_stage/ACC', val_acc * 100, epoch + 1)
            writer.add_scalar('fold_' + str(args.cur_fold) + '_' + 'Val_stage/AUC', val_auc * 100, epoch + 1)
            writer.add_scalar('fold_' + str(args.cur_fold) + '_' + 'Val_stage/SEN', val_sen * 100, epoch + 1)
            writer.add_scalar('fold_' + str(args.cur_fold) + '_' + 'Val_stage/SPE', val_spe * 100, epoch + 1)

            writer.add_scalars('fold_' + str(args.cur_fold) + '_' + 'Test_Val_stage/', {'val_ACC': val_acc * 100, 'test_ACC': test_acc * 100}, epoch + 1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

    if args.save_ckpt:
        filename = save_checkpoint(best_model, best_val_epoch, args, filename='best_ckpt_' + args.session + '_fold_' + str(args.cur_fold) + ".pt")

    return best_model, best_val_acc, best_val_epoch