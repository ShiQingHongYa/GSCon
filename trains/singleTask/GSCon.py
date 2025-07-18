import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
from .KLLoss import KLLoss
from .misc import distance_metric
from thop import profile
import time
logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class Cosine(nn.Module):
    def __init__(self, tau=0.1):
        super(Cosine, self).__init__()
        self.tau = tau

    def forward(self, grad1, grad2):

        norm1 = torch.norm(grad1)
        norm2 = torch.norm(grad2)

        cosine_loss = -torch.sum(grad1 * grad2)/(norm1*norm2*self.tau)

        return cosine_loss

class Relation(nn.Module):
    def __init__(self,tau=0.07, eps=1e-5):
        super(Relation, self).__init__()
        self.tau = tau
        self.eps = eps
        self.l = self.v = self.a =self.length = self.norm_l = self.norm_v = self.norm_a = self.diag = None
        self.relation_l = self.relation_v = self.relation_a =  self.out_l = self.out_v= self.out_a = None

    def forward(self, thi_l, thi_v, thi_a):
        self.l = thi_l
        self.v = thi_v
        self.a = thi_a

        self.length = thi_l.size(0)  # batch_size

        self.norm_l = nn.functional.normalize(self.l, 2, dim=1)
        self.norm_v = nn.functional.normalize(self.v, 2, dim=1)
        self.norm_a = nn.functional.normalize(self.a, 2, dim=1)

        self.diag = torch.eye(self.length).cuda()
        self.relation_l = torch.mm(self.norm_l, self.norm_l.t()) / self.tau
        self.relation_v = torch.mm(self.norm_v, self.norm_v.t()) / self.tau
        self.relation_a = torch.mm(self.norm_a, self.norm_a.t()) / self.tau

        self.relation_l = self.relation_l - self.relation_l * self.diag
        self.relation_v = self.relation_v - self.relation_v * self.diag
        self.relation_a = self.relation_a - self.relation_a * self.diag

        self.out_l = self.relation_l.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length,self.length - 1)  # B*(B-1)
        self.out_v = self.relation_v.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length,self.length - 1)  # B*(B-1)
        self.out_a = self.relation_a.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length,self.length - 1)  # B*(B-1)

        self.out_l = nn.functional.softmax(self.out_l)
        self.out_v = nn.functional.softmax(self.out_v)
        self.out_a = nn.functional.softmax(self.out_a)

        l_v = torch.sum((self.out_l - self.out_v) ** 2, dim=1)
        l_a = torch.sum((self.out_l - self.out_a) ** 2, dim=1)
        v_a = torch.sum((self.out_a - self.out_v) ** 2, dim=1)
        loss = l_v.mean() + l_a.mean() + v_a.mean()

        return loss

class GSCon():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss(reduction='none')
        # self.cosine = Cosine(tau=self.args.ratio_mosi)
        self.norm = nn.Softmax(dim=-1)
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.relation = Relation(tau=0.07, eps=1e-5)  # mosi=0.07    sims=0.06  mosei=0.08



    def do_train(self, model, dataloader, mode='train',return_epoch_results=False):

        params = list(model.parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            start_time= time.time()
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                # single_sample = {}
                for batch_data in td:
                    gradient_samples = {}
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision, is_distill=False)

                    loss_l = self.criterion(output['output_final_l'], labels).cuda()
                    loss_sup_l = loss_l.mean()
                    loss_v = self.criterion(output['output_final_v'], labels).cuda()
                    loss_sup_v = loss_v.mean()
                    loss_a = self.criterion(output['output_final_a'], labels).cuda()
                    loss_sup_a = loss_a.mean()
                    loss_c = self.criterion(output['output_common'], labels).cuda()
                    loss_sup_c = loss_c.mean()

                    loss_sup_lva = loss_sup_l + loss_sup_a + loss_sup_v + loss_sup_c

                    loss_final_max = self.criterion(output['final_output'], labels).mean().cuda()
                    loss_final_cat = self.criterion(output['final_output_three'], labels).mean().cuda()

                    # -----gradient consistent for multi-modality----------------------------------
                    params_to_update = [param for name, param in model.named_parameters() if param.requires_grad and 'common' in name]
                    sum_grad_l = 0
                    sum_grad_v = 0
                    sum_grad_a = 0
                    for i in range(loss_l.size(0)):
                        each_grad_l = torch.autograd.grad(loss_l[i], params_to_update, retain_graph=True,create_graph=True)
                        each_flattened_tensors_l = [t.view(-1) for t in each_grad_l]
                        each_concatenated_l = torch.cat(each_flattened_tensors_l)
                        sum_grad_l = sum_grad_l + each_concatenated_l

                    for i in range(loss_v.size(0)):
                        each_grad_v = torch.autograd.grad(loss_v[i], params_to_update, retain_graph=True,create_graph=True)
                        each_flattened_tensors_v = [t.view(-1) for t in each_grad_v]
                        each_concatenated_v = torch.cat(each_flattened_tensors_v)
                        sum_grad_v = sum_grad_v + each_concatenated_v

                    for i in range(loss_a.size(0)):
                        each_grad_a = torch.autograd.grad(loss_a[i], params_to_update, retain_graph=True,create_graph=True)
                        each_flattened_tensors_a = [t.view(-1) for t in each_grad_a]
                        each_concatenated_a = torch.cat(each_flattened_tensors_a)
                        sum_grad_a = sum_grad_a + each_concatenated_a

                    mean_grad_l = sum_grad_l / loss_l.size(0)
                    mean_grad_v = sum_grad_v / loss_v.size(0)
                    mean_grad_a = sum_grad_a / loss_a.size(0)

                    mean_norm_a = torch.norm(mean_grad_a, dim=0)
                    mean_norm_v = torch.norm(mean_grad_v, dim=0)
                    mean_norm_l = torch.norm(mean_grad_l, dim=0)

                    mean_grad_l = mean_grad_l / mean_norm_l
                    mean_grad_v = mean_grad_v / mean_norm_v
                    mean_grad_a = mean_grad_a / mean_norm_a

                    mean_grad = (mean_grad_a + mean_grad_v + mean_grad_l) / 3

                    dis_grad = -(torch.cosine_similarity(mean_grad_a, mean_grad, dim=0) + torch.cosine_similarity(mean_grad_v, mean_grad, dim=0)
                                + torch.cosine_similarity(mean_grad_l, mean_grad, dim=0)) / 3

                    
                    common_l = output['common_l'].contiguous().view(output['common_l'].size(0), -1)
                    common_v = output['common_v'].contiguous().view(output['common_v'].size(0), -1)
                    common_a = output['common_a'].contiguous().view(output['common_a'].size(0), -1)

                    loss_relation = self.relation(common_l, common_v, common_a)

                    combined_loss = loss_final_cat + loss_final_max + loss_sup_lva + loss_relation + 0.5*dis_grad

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:

                        params = list(model.parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['final_output'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()


            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
                # f">> common_loss: {round(comm_loss, 4)}"
            )
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"模型训练时间: {epoch_time:.4f} 秒")

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")

            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model, self.args.model_save_dir + str(epochs) + '.pth') 
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = self.args.model_save_dir + '/test.pth'
                torch.save(model, model_save_path)   

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop: # or epochs == 30:
                return epoch_results if return_epoch_results else None
            



    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision,is_distill=True)

                    loss1 = self.criterion(output['final_output'], labels).mean()  
                    loss = loss1
                    eval_loss += loss.item()
                    y_pred.append(output['final_output'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results

