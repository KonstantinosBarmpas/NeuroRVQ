import numpy as np
import torch
from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import ch_names_global, create_embedding_ix, check_model_eval_mode
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm

def get_class_weights(y, n_cls):
    y = torch.Tensor(y)
    class_weights = torch.unique(y, return_counts=True)[1]
    class_weights = 1 / class_weights
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights * len(torch.unique(y))  # (n_classes,)
    if len(class_weights) < n_cls:
        tmp = class_weights
        class_weights = torch.zeros(n_cls)
        class_weights[:len(tmp)] = tmp
    class_weights = class_weights.cuda()
    return class_weights

class NeuroRVQModule():
    '''
    Module that performs fine-tuning of NeuroRVQ
    '''
    def __init__(self, sample_length, chnames, n_out, train_head_only, args, foundation_model):
        self.n_time = sample_length // args['patch_size']
        chnames = np.array([c.lower().encode() for c in chnames])
        self.chmask = np.isin(chnames, ch_names_global)
        self.chnames = chnames[self.chmask]
        self.n_out = n_out
        self.model = foundation_model
        self.train_head_only = train_head_only
        self.criterion = F.cross_entropy if self.n_out > 2 else F.binary_cross_entropy_with_logits
        self.results = {'train_accuracy': [], 'val_accuracy': [], 'train_bacc': [], 'val_bacc': []}
        self.weight_decay = args['weight_decay_finetuning']
        self.warmup_epochs = args['warmup_epochs_finetuning']
        self.amp_dtype = torch.bfloat16
        self.lr = float(args['lr_finetuning'])
        self.layer_decay = float(args['layer_decay_finetuning'])
        self.n_patches = args['n_patches']
        self.patch_size = args['patch_size']

    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        d_out = self.n_out if self.n_out > 2 else 1
        self.model.reset_classifier(d_out)
        self.model.cuda()
        # Set model parameter groups with layer_decay on the learning rate
        if self.train_head_only:
            for name, param in self.model.named_parameters():
                if 'head.' in name or 'fc_norm.' in name:
                    continue
                else:
                    param.requires_grad = False

        param_groups = {}
        for i_m, (p_name, param) in enumerate(self.model.named_parameters()):  # model layers
            if not param.requires_grad:
                continue
            if ('head.' in p_name) or ('fc_norm.' in p_name):  # normal lr for classification head
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': self.weight_decay,
                                        'lr': self.lr}
            else:
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': self.weight_decay,
                                        'lr': self.lr * self.layer_decay ** (
                                                len(list(self.model.named_parameters())) - i_m)}

        # Optimizer and lr_scheduler
        optimizer = torch.optim.AdamW(list(param_groups.values()))
        n_batches_train = int(np.ceil(len(train_dataset) / batch_size))
        if epochs < self.warmup_epochs + 1:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, end_factor=1,
                                                             total_iters=epochs * n_batches_train)
        else:
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, end_factor=1,
                                                           total_iters=self.warmup_epochs * n_batches_train)
            scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-1,
                                                           total_iters=(epochs - self.warmup_epochs) * n_batches_train)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2],
                                                                 milestones=[self.warmup_epochs * n_batches_train])
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')
        # Prepare automatic mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        y_train = [ys for _, ys in train_dataset]
        y_val = [ys for _, ys in validation_dataset]
        y = y_train + y_val
        class_weights = get_class_weights(y, self.n_out)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        temp_embed_ix, spat_embed_ix = create_embedding_ix(self.n_time, self.n_patches,
                                                           self.chnames, ch_names_global)

        # Loop over epochs
        for i_epoch in range(epochs):
            print(f"Epoch {i_epoch}")
            # Loop over training batches
            self.model.train()
            e_pred_train = []  # collect predictions
            y_true_train = []  # y in order seen
            for x_b, y_b in tqdm(train_dataloader):
                x_b = x_b[:, self.chmask, :]
                n, c, t = x_b.shape
                x_b = x_b.reshape(n, c, self.n_time, self.patch_size).cuda()
                y_b = y_b.long() if self.n_out > 2 else y_b.float()
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    optimizer.zero_grad()
                    p, _ = self.model(x_b, temp_embed_ix, spat_embed_ix)
                    p = p.squeeze(-1)  # remove class dim if binary task
                    loss_weight = class_weights if p.ndim == 2 else class_weights[y_b.long()]
                    loss = self.criterion(p, y_b.cuda(), weight=loss_weight)


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                # Collect class predictions to compute metrics on the full epoch
                p = p.detach().cpu().float()
                p = p.argmax(dim=-1) if p.ndim == 2 else torch.round(torch.sigmoid(p))
                e_pred_train += [p.numpy()]
                y_true_train += [y_b.numpy()]

            # Loop over validation batches
            self.model.eval()
            e_pred_val = []  # collect predictions
            y_true_val = []  # y in order seen
            for x_b, y_b in tqdm(val_dataloader):
                x_b = x_b[:, self.chmask, :]
                n, c, t = x_b.shape
                x_b = x_b.reshape(n, c, self.n_time, self.patch_size).cuda()
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    p, _ = self.model(x_b, temp_embed_ix, spat_embed_ix)
                    p = p.squeeze(-1)  # remove class dim if binary task

                # Collect class predictions to compute metrics on the full epoch
                p = p.detach().cpu().float()
                p = p.argmax(dim=-1) if p.ndim == 2 else torch.round(torch.sigmoid(p))
                e_pred_val += [p.numpy()]
                y_true_val += [y_b.numpy()]

            # Compute accuracy and balanced accuracy
            e_pred_train = np.concatenate(e_pred_train)
            e_pred_val = np.concatenate(e_pred_val)
            y_true_train = np.concatenate(y_true_train)
            y_true_val = np.concatenate(y_true_val)

            self.results['train_accuracy'] += [accuracy_score(y_true_train, e_pred_train)]
            self.results['val_accuracy'] += [accuracy_score(y_true_val, e_pred_val)]
            self.results['train_bacc'] += [balanced_accuracy_score(y_true_train, e_pred_train)]
            self.results['val_bacc'] += [balanced_accuracy_score(y_true_val, e_pred_val)]
            if len(validation_dataset) > 1:
                print(f"VAL ACC: {self.results['val_accuracy'][-1]}, VAL BACC: {self.results['val_bacc'][-1]}")
