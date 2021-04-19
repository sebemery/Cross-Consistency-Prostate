import torch
import time, random, cv2, sys 
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainerMulti
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import scale


class Trainer(BaseTrainerMulti):

    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch, self.supervised_loader.dataset.site, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode

        self.start_time = time.time()

    def _train_epoch(self, epoch):
        self.html_results.save()

        self.logger.info('\n')
        self.model.train()

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135)
        else:
            dataloader = iter(zip(self.supervised_loader, cycle(self.unsupervised_loader)))
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135)

        self._reset_metrics()
        for batch_idx in tbar:
            if self.mode == 'supervised':
                (input_l, target_l), (input_ul, target_ul) = next(dataloader), (None, None)
            else:
                (input_l, target_l), (input_ul, target_ul) = next(dataloader)
                if self.str_device == 'gpu':
                    input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)

            if self.str_device == 'gpu':
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            self.optimizer.zero_grad()

            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                         curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)
            self.count += input_l.size(0)
            logs = self._log_values(cur_losses)

            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                self._write_scalars_tb(logs)

            if batch_idx % int(len(self.unsupervised_loader) * 0.9) == 0:
                self._write_img_tb(input_l, target_l, input_ul, target_ul, outputs, epoch)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            tbar.set_description('T ({}) | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f} DSL {:.2f} DSUL {:.2f}|'.format(epoch, self.loss_sup.average, self.loss_unsup.average,self.loss_weakly.average, self.pair_wise.average, self.mIoU_l,self.mIoU_ul, self.mdice_l, self.mdice_ul))

            self.lr_scheduler.step(epoch=epoch - 1)

        return logs

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        total_dice = 0
        count = 0

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                if self.str_device == 'gpu':
                    target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                output = self.model(data)

                # LOSS
                loss = F.cross_entropy(output, target)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union, dice = eval_metrics(output, target, self.num_classes, self.str_device)
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled
                total_dice = ((count * total_dice) + (dice * output.size(0))) / (count + output.size(0))
                count += output.size(0)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    if isinstance(data, list): data = data[0]
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                mdice = dice.mean()
                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                               "Mean_dice": np.round(mdice, 3),
                               "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
                               "Class_dice": dict(zip(range(self.num_classes), np.round(dice, 3)))}

                tbar.set_description(
                    'EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} Mean Dice {:.2f} |'.format(epoch,
                                                                                                             total_loss_val.average,
                                                                                                             pixAcc,
                                                                                                             mIoU,
                                                                                                             mdice))

            self._add_img_tb(val_visual, 'val')

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
            for k, v in list(seg_metrics.items())[:-2]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }
            self.html_results.add_results(epoch=epoch, seg_resuts=log)
            self.html_results.save()

            if (time.time() - self.start_time) / 3600 > 22:
                self._save_checkpoint(epoch, save_best=self.improved)
        return log

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.dice_l, self.dice_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.mdice_l, self.mdice_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.class_dice_l, self.class_dice_ul = {}, {}
        self.count = 0

    def _update_losses(self, cur_losses):
        if "loss_sup" in cur_losses.keys():
            self.loss_sup.update(cur_losses['loss_sup'].mean().item())
        if "loss_unsup" in cur_losses.keys():
            self.loss_unsup.update(cur_losses['loss_unsup'].mean().item())
        if "loss_weakly" in cur_losses.keys():
            self.loss_weakly.update(cur_losses['loss_weakly'].mean().item())
        if "pair_wise" in cur_losses.keys():
            self.pair_wise.update(cur_losses['pair_wise'].mean().item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.str_device)
        self._update_seg_metrics(*seg_metrics_l, target_l.size(0), True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.pixel_acc_l, self.mIoU_l, self.class_iou_l, self.mdice_l, self.class_dice_l = seg_metrics_l.values()

        if self.mode == 'semi':
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.str_device)
            self._update_seg_metrics(*seg_metrics_ul, target_ul.size(0), False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul, self.mdice_ul, self.class_dice_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, dice, batch_size, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
            self.dice_l = ((self.dice_l * self.count) + (dice * batch_size)) / (self.count + batch_size)
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union
            self.dice_ul = ((self.dice_ul * self.count) + (dice * batch_size)) / (self.count + batch_size)

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
            dice = self.dice_l
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
            dice = self.dice_ul
        mIoU = IoU.mean()
        mdice = dice.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
            "Mean_dice": np.round(mdice, 3),
            "Class_dice": dict(zip(range(self.num_classes), np.round(dice, 3)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        logs['dice_score_labeled'] = self.mdice_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
            logs['dice_score_unlabeled'] = self.mdice_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        current_rampup = self.model.module.unsup_loss_w.current_rampup
        self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)

    def denormalize(self, image, mean, std):
        image = (image * std) + mean
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        return image

    def _add_img_tb(self, val_visual, wrt_mode):

        if wrt_mode == 'supervised':
            mean = self.supervised_loader.MEAN
            std = self.supervised_loader.STD
        if wrt_mode == 'unsupervised':
            mean = self.unsupervised_loader.MEAN
            std = self.unsupervised_loader.STD
        if wrt_mode == 'val':
            mean = self.val_loader.MEAN
            std = self.val_loader.STD

        val_img = []
        for imgs in val_visual:
            imgs = [self.denormalize(i, mean, std) if (isinstance(i, torch.Tensor)) else scale(i) for i in imgs]
            imgs = [i if (isinstance(i, torch.Tensor)) else torch.from_numpy(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)  # [3*batch_size,1,384,384]
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step, dataformats='CHW')

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        if self.mode == 'semi':
            outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
            targets_ul_np = target_ul.data.cpu().numpy()
            imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul, outputs_ul_np, targets_ul_np)]
            self._add_img_tb(imgs, 'unsupervised')


class Multi_Trainer:

    def __init__(self, model_1,model_2,model_3,model_4,model_5,model_6, resume1,resume2,resume3,resume4,resume5,resume6,
                 config, supervised_loader_1, unsupervised_loader_1, iter_per_epoch_1,supervised_loader_2,
                 unsupervised_loader_2, iter_per_epoch_2,supervised_loader_3, unsupervised_loader_3, iter_per_epoch_3,
                 supervised_loader_4, unsupervised_loader_4, iter_per_epoch_4,supervised_loader_5, unsupervised_loader_5,
                 iter_per_epoch_5,supervised_loader_6, unsupervised_loader_6, iter_per_epoch_6,val_loader_1=None,
                 val_loader_2=None,val_loader_3=None,val_loader_4=None,val_loader_5=None,val_loader_6=None,
                 train_logger_1=None,train_logger_2=None,train_logger_3=None,train_logger_4=None,train_logger_5=None,
                 train_logger_6=None):

        # Trainer for individual domain
        self.trainer_multi_1 = Trainer(model=model_1, resume=resume1, config=config, supervised_loader=supervised_loader_1,
                                       unsupervised_loader=unsupervised_loader_1,val_loader=val_loader_1,
                                       iter_per_epoch=iter_per_epoch_1,train_logger=train_logger_1)
        self.trainer_multi_2 = Trainer(model=model_2,resume=resume2,config=config,supervised_loader=supervised_loader_2,
                                       unsupervised_loader=unsupervised_loader_2, val_loader=val_loader_2,
                                       iter_per_epoch=iter_per_epoch_2,train_logger=train_logger_2)
        self.trainer_multi_3 = Trainer(model=model_3, resume=resume3, config=config,
                                       supervised_loader=supervised_loader_3,
                                       unsupervised_loader=unsupervised_loader_3, val_loader=val_loader_3,
                                       iter_per_epoch=iter_per_epoch_3,train_logger=train_logger_3)
        self.trainer_multi_4 = Trainer(model=model_4, resume=resume4, config=config,
                                       supervised_loader=supervised_loader_4,
                                       unsupervised_loader=unsupervised_loader_4, val_loader=val_loader_4,
                                       iter_per_epoch=iter_per_epoch_4,train_logger=train_logger_4)
        self.trainer_multi_5 = Trainer(model=model_5, resume=resume5, config=config,
                                       supervised_loader=supervised_loader_5,
                                       unsupervised_loader=unsupervised_loader_5, val_loader=val_loader_5,
                                       iter_per_epoch=iter_per_epoch_5,train_logger=train_logger_5)
        self.trainer_multi_6 = Trainer(model=model_6, resume=resume6, config=config,
                                       supervised_loader=supervised_loader_6,
                                       unsupervised_loader=unsupervised_loader_6, val_loader=val_loader_6,
                                       iter_per_epoch=iter_per_epoch_6,train_logger=train_logger_6)
        # nb epochs
        self.start_epoch = 1
        self.epochs = config['trainer']['epochs']
        self.do_validation = config['trainer']['val']
        self.val_per_epochs = config['trainer']['val_per_epochs']

        if resume1:
            self._resume_checkpoint_multi(resume1)

    def train(self):
        # train alternatively all domain at each iteration
        for epoch in range(self.start_epoch, self.epochs + 1):
            # call each domain trainer
            self.trainer_multi_1.train(epoch)
            self.trainer_multi_2.train(epoch)
            self.trainer_multi_3.train(epoch)
            self.trainer_multi_4.train(epoch)
            self.trainer_multi_5.train(epoch)
            self.trainer_multi_6.train(epoch)
            # validation for each domain trainer
            if self.do_validation and epoch % self.val_per_epochs == 0:
                self.trainer_multi_1.validation(epoch)
                self.trainer_multi_2.validation(epoch)
                self.trainer_multi_3.validation(epoch)
                self.trainer_multi_4.validation(epoch)
                self.trainer_multi_5.validation(epoch)
                self.trainer_multi_6.validation(epoch)

        # save results
        self.trainer_multi_1.html_results.save()
        self.trainer_multi_2.html_results.save()
        self.trainer_multi_3.html_results.save()
        self.trainer_multi_4.html_results.save()
        self.trainer_multi_5.html_results.save()
        self.trainer_multi_6.html_results.save()
        self.trainer_multi_1.writer.flush()
        self.trainer_multi_1.writer.close()
        self.trainer_multi_2.writer.flush()
        self.trainer_multi_2.writer.close()
        self.trainer_multi_3.writer.flush()
        self.trainer_multi_3.writer.close()
        self.trainer_multi_4.writer.flush()
        self.trainer_multi_4.writer.close()
        self.trainer_multi_5.writer.flush()
        self.trainer_multi_5.writer.close()
        self.trainer_multi_6.writer.flush()
        self.trainer_multi_6.writer.close()

    def _resume_checkpoint_multi(self, resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
