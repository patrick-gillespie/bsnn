#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# Modifications 2024 Patrick Gillespie
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from models.bayes_disc_models import BayesDiagSheafDiffusion, BayesBundleSheafDiffusion, BayesGeneralSheafDiffusion, GCN
from utils.heterophilic import get_dataset, get_fixed_splits


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data, args, epoch):
    model.train()
    optimizer.zero_grad()

    # beta is scalar used to cyclically anneal kl loss term
    beta = torch.sigmoid(torch.tensor((epoch % 40) / 2 - 10))

    if args.bayes_model:
        out, kl = model(data.x)
        out = out[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        if args.use_kl:
            loss = nll + beta * kl
        else:
            loss = nll

    else:
        out = model(data.x)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll

    loss.backward()
    optimizer.step()
    del out


def test(model, data, args):
    model.eval()
    with torch.no_grad():
        if args.bayes_model:
            probs_list = []
            for i in range(args.num_ensemble):
                logits, kl = model(data.x)
                probs_list.append(torch.exp(logits))
            probs = torch.mean(torch.stack(probs_list), 0, keepdim=True).squeeze(0)
            logits = torch.log(probs)
        else:
            logits = model(data.x)
            kl = None

        accs, losses, preds = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses, kl


def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold, args['permute_masks'])
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data, args, epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss], kl = test(model, data, args)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
                f'fold{fold}_kl': kl,
            }
            wandb.log(res_dict, step=epoch)

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    # if "ODE" not in args['model']:
    #     # Debugging for discrete models
    #     for i in range(len(model.sheaf_learners)):
    #         L_max = model.sheaf_learners[i].L.detach().max().item()
    #         L_min = model.sheaf_learners[i].L.detach().min().item()
    #         L_avg = model.sheaf_learners[i].L.detach().mean().item()
    #         L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
    #         print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")
    #
    #     with np.printoptions(precision=3, suppress=True):
    #         for i in range(0, args['layers']):
    #             print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, keep_running


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if args.model == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args.model == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args.model == 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    elif args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    elif args.model == 'BayesDiagSheaf':
        model_cls = BayesDiagSheafDiffusion
    elif args.model == 'BayesBundleSheaf':
        model_cls = BayesBundleSheafDiffusion
    elif args.model == 'BayesGeneralSheaf':
        model_cls = BayesGeneralSheafDiffusion
    elif args.model == 'GCN':
        model_cls = GCN
    else:
        raise ValueError(f'Unknown model {args.model}')

    dataset = get_dataset(args.dataset)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # Add extra arguments
    args.bayes_model = 'Bayes' in args.model
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)
    wandb.init(project="sheaf", config=vars(args), entity=args.entity)

    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        if not keep_running:
            break

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')

