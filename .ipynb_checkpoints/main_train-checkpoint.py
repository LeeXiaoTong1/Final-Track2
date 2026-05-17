import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import shutil
import numpy as np
from sklearn.metrics import f1_score

from model import *
from dataset import *
from CSAM import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler
from backbone.rawaasist import *
from collections import defaultdict
from tqdm import tqdm, trange
from exp.feature_extraction_exp import *
from utils import *
import eval_metrics as em
from feature_extraction import *
import config

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)


def initParams():
    parser = config.initParams()

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=4, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")

    parser.add_argument('--train_task', type=str, default="atadd-track1",
                        choices=["atadd-track1", "atadd-track2"])
    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--continue_training', action='store_true',
                        help="continue training with trained model")

    parser.add_argument(
        '--save_best_by',
        type=str,
        default='loss',
        choices=['loss', 'eer', 'f1'],
        help='Metric used to save the best model: loss, eer, or f1'
    )
    
    # generalized strategy
    parser.add_argument('--SAM', type=bool, default=False, help="use SAM")
    parser.add_argument('--ASAM', type=bool, default=False, help="use ASAM")
    parser.add_argument('--CSAM', type=bool, default=False, help="use CSAM")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            json.dump(vars(args), file, indent=4)

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("epoch\tstep\ttrain_loss\n")

        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("epoch\tval_loss\tval_eer\tval_f1\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
def unpack_batch(batch, device):
    """
    Support both:
    Track1 / original: waveform, filename, label
    Track2 type-aware: waveform, filename, label, type_id
    """
    if len(batch) == 4:
        feat, audio_fn, labels, type_ids = batch
        type_ids = type_ids.to(device, non_blocking=True).long()
    else:
        feat, audio_fn, labels = batch
        type_ids = None

    feat = feat.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()

    return feat, audio_fn, labels, type_ids


def type_group_dro_ce_loss(outputs, labels, type_ids, class_weight=None, eta=2.0, n_types=4):
    """
    Type-balanced GroupDRO:
    compute CE loss per audio type and emphasize the worst type.
    """
    sample_losses = F.cross_entropy(
        outputs,
        labels,
        weight=class_weight,
        reduction="none"
    )

    group_losses = []
    valid_types = []

    for t in range(n_types):
        mask = (type_ids == t)
        if mask.any():
            group_losses.append(sample_losses[mask].mean())
            valid_types.append(t)

    if len(group_losses) == 0:
        return sample_losses.mean()

    group_losses = torch.stack(group_losses)
    group_weights = torch.softmax(eta * group_losses.detach(), dim=0)

    loss = (group_weights * group_losses).sum()
    return loss


def track2_macro_f1_by_type(labels, preds, type_ids, n_types=4):
    """
    Official-like Track2 dev metric:
    first compute Macro-F1 within each type, then average four types.
    """
    type_f1s = []

    for t in range(n_types):
        mask = (type_ids == t)
        if mask.sum() == 0:
            continue

        f1_t = f1_score(
            labels[mask],
            preds[mask],
            average="macro",
            labels=[0, 1],
            zero_division=0
        )
        type_f1s.append(f1_t)

    if len(type_f1s) == 0:
        return f1_score(labels, preds, average="macro", zero_division=0), []

    return float(np.mean(type_f1s)), type_f1s


def router_entropy_loss(expert_weights):
    """
    Encourage router not to collapse to a single expert too early.
    Return negative entropy loss term usage:
    loss = loss - weight * entropy
    """
    entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=-1).mean()
    return entropy


def find_bad_grads(model, max_print=30):
    bad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                g = p.grad.detach()
                bad.append(name)
                print(
                    f"[BAD GRAD] {name} | "
                    f"shape={tuple(g.shape)} | "
                    f"nan={torch.isnan(g).any().item()} | "
                    f"inf={torch.isinf(g).any().item()} | "
                    f"min={torch.nan_to_num(g).min().item():.4e} | "
                    f"max={torch.nan_to_num(g).max().item():.4e}"
                )
                if len(bad) >= max_print:
                    break
    return bad


def find_bad_params(model, max_print=30):
    bad = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if not torch.isfinite(p).all():
                bad.append(name)
                x = p.detach()
                print(
                    f"[BAD PARAM] {name} | "
                    f"shape={tuple(x.shape)} | "
                    f"nan={torch.isnan(x).any().item()} | "
                    f"inf={torch.isinf(x).any().item()} | "
                    f"min={torch.nan_to_num(x).min().item():.4e} | "
                    f"max={torch.nan_to_num(x).max().item():.4e}"
                )
                if len(bad) >= max_print:
                    break
    return bad

####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT

def shuffle(feat, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    if args.train_task == "atadd-track2" and (args.t2_gdro or args.t2_type_adv) and (args.SAM or args.ASAM or args.CSAM):
        raise ValueError("Do not mix Track2 GDRO/type-adversarial training with SAM/ASAM/CSAM in the first version.")
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT

    # initialize model
    # Conventional CM 
    if args.model == 'aasist':
        feat_model = Rawaasist().to(args.device)
    if args.model == 'specresnet':
        feat_model = ResNet18ForAudio().to(args.device)
    #❄️ FR SSL-based CM
    if args.model == 'fr-w2v2aasist':
        feat_model = XLSRAASIST(model_dir=args.xlsr).to(args.device)
    if args.model == 'fr-wavlmaasist':
        feat_model = WAVLMAASIST(model_dir=args.wavlm).to(args.device)
    if args.model == 'fr-mertaasist':
        feat_model = MERTAASIST(model_dir=args.mert).to(args.device)
    #🔥 FT-SSL-based CM
    if args.model == 'ft-w2v2aasist':
        feat_model = XLSRAASIST(model_dir=args.xlsr, freeze=False).to(args.device)
    if args.model == 'ft-wavlmaasist':
        feat_model = WAVLMAASIST(model_dir=args.wavlm, freeze=False).to(args.device)
    if args.model == 'ft-mertaasist':
        feat_model = MERTAASIST(model_dir=args.mert, freeze=False).to(args.device)
    if args.model == "ufm-track2-full":
        feat_model = UFMTrack2Full(
            xlsr_dir=args.xlsr,
            mert_dir=args.mert,
            beats_dir=args.beats,
            device=args.device,
            freeze_xlsr=args.ufm_freeze_xlsr,
            freeze_mert=args.ufm_freeze_mert,
            freeze_beats=args.ufm_freeze_beats,
            dim=args.ufm_dim,
            mem_slots=args.ufm_mem_slots,
            heads=args.ufm_heads,
            layers=args.ufm_layers,
            dropout=args.ufm_dropout
        ).to(args.device)
        
    if getattr(args, "init_from", ""):
        print(f"Loading initialization checkpoint from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=args.device)
        missing, unexpected = feat_model.load_state_dict(ckpt, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    if args.model == 't2-router-xlsr-mert':
        feat_model = TypeRoutedXLSRMERTAASIST(
            xlsr_dir=args.xlsr,
            mert_dir=args.mert,
            device=args.device,
            freeze_xlsr=args.t2_router_freeze_xlsr,
            freeze_mert=args.t2_router_freeze_mert
        ).to(args.device)
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    #🔥 WPT-SSL-based CM 
    if args.model == 'pt-w2v2aasist':
        feat_model = PTW2V2AASIST(
            model_dir=args.xlsr,
            prompt_dim=args.prompt_dim,
            num_prompt_tokens=args.num_prompt_tokens,
            dropout=args.pt_dropout
        ).to(args.device)
    if args.model == "wpt-w2v2aasist":
        feat_model = WPTW2V2AASIST(
            model_dir=args.xlsr,
            prompt_dim=args.prompt_dim,
            num_prompt_tokens=args.num_prompt_tokens,
            num_wavelet_tokens=args.num_wavelet_tokens,
            dropout=args.pt_dropout
        ).to(args.device)
        
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    if getattr(args, "t2_type_adv", False):
        feat_model.type_adv_head = TypeHead(
            in_dim=args.t2_type_feat_dim,
            n_types=4,
            hidden_dim=128,
            dropout=0.1
        ).to(args.device)
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    
    feat_optimizer = torch.optim.Adam(
        feat_model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005
    )

    if args.SAM or args.CSAM:
        base_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            base_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    if args.train_task == "atadd-track1":
        atadd_t1_trainset = atadd_dataset(
            args.atadd_t1_train_audio,
            args.atadd_t1_train_label,
            audio_length=args.audio_len
        )
        atadd_t1_devset = atadd_dataset(
            args.atadd_t1_dev_audio,
            args.atadd_t1_dev_label,
            audio_length=args.audio_len
        )
        train_set = [atadd_t1_trainset]
        dev_set = [atadd_t1_devset]

    # if args.train_task == "atadd-track2":
    #     atadd_t2_trainset = atadd_dataset(
    #         args.atadd_t2_train_audio,
    #         args.atadd_t2_train_label,
    #         audio_length=args.audio_len
    #     )
    #     atadd_t2_devset = atadd_dataset(
    #         args.atadd_t2_dev_audio,
    #         args.atadd_t2_dev_label,
    #         audio_length=args.audio_len
    #     )
    #     train_set = [atadd_t2_trainset]
    #     dev_set = [atadd_t2_devset]

    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    if args.train_task == "atadd-track2":
        need_type_id = (
            args.t2_return_type or
            args.t2_gdro or
            args.t2_type_adv or
            args.t2_router_type_loss > 0 or
            getattr(args, "ufm_type_loss", 0) > 0
        )

        atadd_t2_trainset = atadd_dataset(
            args.atadd_t2_train_audio,
            args.atadd_t2_train_label,
            audio_length=args.audio_len,
            return_type=need_type_id
        )

        atadd_t2_devset = atadd_dataset(
            args.atadd_t2_dev_audio,
            args.atadd_t2_dev_label,
            audio_length=args.audio_len,
            return_type=need_type_id
        )

        train_set = [atadd_t2_trainset]
        dev_set = [atadd_t2_devset]
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    
    for dataset in train_set:
        print(len(dataset), f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."
    for dataset in dev_set:
        print(len(dataset), f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."

    training_set = ConcatDataset(train_set)
    validation_set = ConcatDataset(dev_set)

    trainOriDataLoader = DataLoader(
        training_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))),
        pin_memory=args.cuda
    )

    valOriDataLoader = DataLoader(
        validation_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))),
        pin_memory=args.cuda
    )

    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)

    if args.train_task == "atadd-track1":
        weight = torch.FloatTensor([4, 1]).to(args.device)

    if args.train_task == "atadd-track2":
        weight = torch.FloatTensor([3.5, 1]).to(args.device)

    print(f"Using class weight: {weight.tolist()}")
    print(f"Best model will be saved by: {args.save_best_by}")

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    prev_loss = float("inf")
    prev_eer = float("inf")
    prev_f1 = -float("inf")
    monitor_loss = 'base_loss'

    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)

        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
            try:
                batch = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                batch = next(trainOri_flow)

            feat, audio_fn, labels, type_ids = unpack_batch(batch, args.device)
            ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
            
#             try:
#                 feat, audio_fn, labels = next(trainOri_flow)
#             except StopIteration:
#                 trainOri_flow = iter(trainOriDataLoader)
#                 feat, audio_fn, labels = next(trainOri_flow)

#             feat = feat.to(args.device, non_blocking=True)
#             labels = labels.to(args.device, non_blocking=True)

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()
                feat_optimizer.second_step(zero_grad=True)

            # else:
            #     feat_optimizer.zero_grad()
            #     feats, feat_outputs = feat_model(feat)
            #     feat_loss = criterion(feat_outputs, labels)
            #     feat_loss.backward()
            #     feat_optimizer.step()
            
            ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
            else:
                feat_optimizer.zero_grad()

                feats, feat_outputs = feat_model(feat)

                # Detection loss
                if args.train_task == "atadd-track2" and args.t2_gdro and type_ids is not None:
                    feat_loss = type_group_dro_ce_loss(
                        outputs=feat_outputs,
                        labels=labels,
                        type_ids=type_ids,
                        class_weight=weight,
                        eta=args.t2_gdro_eta,
                        n_types=4
                    )
                else:
                    feat_loss = criterion(feat_outputs, labels)

                # Type-adversarial loss: make feature less type-specific
                if (
                    args.train_task == "atadd-track2"
                    and args.t2_type_adv
                    and type_ids is not None
                    and hasattr(feat_model, "type_adv_head")
                ):
                    z = pool_for_type_head(feats)
                    type_logits_adv = feat_model.type_adv_head(
                        grad_reverse(z, args.t2_grl_lambda)
                    )
                    loss_type_adv = F.cross_entropy(type_logits_adv, type_ids)
                    feat_loss = feat_loss + args.t2_type_adv_weight * loss_type_adv

                # Router type supervision: make router aware of audio type during training
                if (
                    args.train_task == "atadd-track2"
                    and args.t2_router_type_loss > 0
                    and type_ids is not None
                    and hasattr(feat_model, "latest_type_logits")
                    and feat_model.latest_type_logits is not None
                ):
                    loss_router_type = F.cross_entropy(
                        feat_model.latest_type_logits,
                        type_ids
                    )
                    feat_loss = feat_loss + args.t2_router_type_loss * loss_router_type

                # Router entropy: avoid expert collapse
                if (
                    args.train_task == "atadd-track2"
                    and args.t2_router_entropy > 0
                    and hasattr(feat_model, "latest_expert_weights")
                    and feat_model.latest_expert_weights is not None
                ):
                    ent = router_entropy_loss(feat_model.latest_expert_weights)
                    feat_loss = feat_loss - args.t2_router_entropy * ent

                # ======================================================
                # Optional: UFM auxiliary type loss
                # ======================================================
                if (
                    args.train_task == "atadd-track2"
                    and getattr(args, "ufm_type_loss", 0) > 0
                    and type_ids is not None
                    and hasattr(feat_model, "latest_type_logits")
                    and feat_model.latest_type_logits is not None
                ):
                    loss_ufm_type = F.cross_entropy(
                        feat_model.latest_type_logits,
                        type_ids
                    )

                    feat_loss = feat_loss + args.ufm_type_loss * loss_ufm_type

                # ======================================================
                # Optional: UFM router entropy regularization
                # ======================================================
                if (
                    args.train_task == "atadd-track2"
                    and getattr(args, "ufm_router_entropy", 0) > 0
                    and hasattr(feat_model, "latest_expert_weights")
                    and feat_model.latest_expert_weights is not None
                ):
                    ent = router_entropy_loss(
                        feat_model.latest_expert_weights
                    )

                    feat_loss = feat_loss - args.ufm_router_entropy * ent
                    
                if not torch.isfinite(feat_loss):
                    print(
                        f"[skip non-finite loss] epoch={epoch_num}, step={i}, "
                        f"loss={feat_loss.item()}"
                    )
                    feat_optimizer.zero_grad(set_to_none=True)
                    continue

                feat_loss.backward()

                bad_grads = find_bad_grads(feat_model)

                if len(bad_grads) > 0:
                    print(f"[STOP] non-finite gradients at epoch={epoch_num}, step={i}")
                    feat_optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(f"Non-finite gradients found: {bad_grads[:10]}")

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    feat_model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=False
                )

                if not torch.isfinite(grad_norm):
                    print(
                        f"[skip non-finite grad_norm] epoch={epoch_num}, step={i}, "
                        f"grad_norm={grad_norm}"
                    )
                    feat_optimizer.zero_grad(set_to_none=True)
                    continue

                feat_optimizer.step()

                bad_params = find_bad_params(feat_model)

                if len(bad_params) > 0:
                    print(f"[STOP] non-finite parameters after step epoch={epoch_num}, step={i}")
                    raise RuntimeError(f"Non-finite parameters found: {bad_params[:10]}")

            

            trainlossDict['base_loss'].append(feat_loss.item())

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(
                    str(epoch_num) + "\t" +
                    str(i) + "\t" +
                    str(trainlossDict[monitor_loss][-1]) + "\n"
                )

        feat_model.eval()
        with torch.no_grad():
            # ip1_loader, tag_loader, idx_loader, score_loader, pred_loader = [], [], [], [], []
            ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
            ip1_loader, tag_loader, idx_loader, score_loader, pred_loader, type_loader = [], [], [], [], [], []
            ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT

            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
#                 try:
#                     feat, audio_fn, labels = next(valOri_flow)
#                 except StopIteration:
#                     valOri_flow = iter(valOriDataLoader)
#                     feat, audio_fn, labels = next(valOri_flow)

#                 feat = feat.to(args.device, non_blocking=True)
#                 labels = labels.to(args.device, non_blocking=True)
                
                ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
                try:
                    batch = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    batch = next(valOri_flow)

                feat, audio_fn, labels, type_ids = unpack_batch(batch, args.device)
                ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT

                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = torch.sigmoid(feat_outputs[:, 0])
                    pred = torch.where(score >= 0.5,
                                       torch.zeros_like(labels),
                                       torch.ones_like(labels))
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    prob = F.softmax(feat_outputs, dim=1)
                    score = prob[:, 0]   
                    pred = torch.where(score >= 0.5,
                                       torch.zeros_like(labels),
                                       torch.ones_like(labels))

                ip1_loader.append(feats)
                idx_loader.append(labels)
                pred_loader.append(pred)
                devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)
                
                ### 5,13
                if type_ids is not None:
                    type_loader.append(type_ids)
                ### 5,13
                
                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '


            valLoss = np.nanmean(devlossDict[monitor_loss])
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            preds = torch.cat(pred_loader, 0).data.cpu().numpy()

            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            # val_f1 = f1_score(labels, preds, average='macro')
            
            ### 5,13
            if args.train_task == "atadd-track2" and len(type_loader) > 0:
                type_ids_np = torch.cat(type_loader, 0).data.cpu().numpy()
                val_f1, type_f1s = track2_macro_f1_by_type(
                    labels=labels,
                    preds=preds,
                    type_ids=type_ids_np,
                    n_types=4
                )
                print("Track2 Type F1s [speech, sound, singing, music]:", type_f1s)
            else:
                val_f1 = f1_score(labels, preds, average='macro', zero_division=0)
            ### 5,13
            
            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(
                    str(epoch_num) + "\t" +
                    str(valLoss) + "\t" +
                    str(val_eer) + "\t" +
                    str(val_f1) + "\n"
                )

            print("Val Loss: {}".format(valLoss))
            print("Val EER: {}".format(val_eer))
            print("Val F1 : {}".format(val_f1))

        if (epoch_num + 1) % 5 == 0:
            torch.save(
                feat_model.state_dict(),
                os.path.join(args.out_fold, 'checkpoint', 'atadd_model_%d.pt' % (epoch_num + 1))
            )

        save_flag = False

        if args.save_best_by == "loss":
            if valLoss < prev_loss:
                prev_loss = valLoss
                save_flag = True

        elif args.save_best_by == "eer":
            if val_eer < prev_eer:
                prev_eer = val_eer
                save_flag = True

        elif args.save_best_by == "f1":
            if val_f1 > prev_f1:
                prev_f1 = val_f1
                save_flag = True

        if save_flag:
            torch.save(
                feat_model.state_dict(),
                os.path.join(args.out_fold, 'atadd_model.pt')
            )
            print(f"Best model updated by {args.save_best_by} at epoch {epoch_num}")

    return feat_model


if __name__ == "__main__":
    args = initParams()
    train(args)
