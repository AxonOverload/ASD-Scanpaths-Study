import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score

from .preprocessing import all_data_filtered
from .features import build_feature_sequence
from .models import collate_fn, LSTM, GRU






def run_fold(train_entries, test_entries, cache, shuffle_fixations=False, use_duration=True,
             model_class=LSTM, hidden_size=64, epochs=30):

    def build(entries):
        data = []
        for e in entries:
            seq = build_feature_sequence(e, cache)
            if not use_duration:
                seq = seq[:, :512]
            if shuffle_fixations:
                perm = torch.randperm(seq.shape[0])
                seq = seq[perm]
            label = 1 if e['label'] == 'ASD' else 0
            data.append((seq, label))
        return data


    train_dataset = build(train_entries)
    test_dataset = build(test_entries)


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)



    input_size = 512 if not use_duration else 513
    model = model_class(input_size=input_size, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()


    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for padded_seqs, lengths, labels_batch in train_loader:
            optimizer.zero_grad()
            logits = model(padded_seqs, lengths)
            loss = criterion(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = running_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for padded_seqs, lengths, labels_batch in test_loader:
            logits = model(padded_seqs, lengths)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels_batch.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), epoch_losses




labels = [1 if e['label'] == 'ASD' else 0 for e in all_data_filtered]
groups = [e['participant_id'] for e in all_data_filtered]



sgkf = StratifiedGroupKFold(n_splits=7)


def run_cv(all_data_filtered,labels,groups,cache,config_name, **kwargs):
    accs, precs, recs, f1s = [], [], [], []
    fold_losses = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(all_data_filtered, labels, groups=groups)):
        train_entries = [all_data_filtered[i] for i in train_idx]
        test_entries = [all_data_filtered[i] for i in test_idx]

        preds, targets, losses = run_fold(train_entries, test_entries, cache, **kwargs)
        fold_losses.append(losses)

        accs.append((preds == targets).mean())
        precs.append(precision_score(targets, preds, zero_division=0))
        recs.append(recall_score(targets, preds, zero_division=0))
        f1s.append(f1_score(targets, preds, zero_division=0))

    metrics = {
        'acc': (np.mean(accs), np.std(accs)),
        'prec': (np.mean(precs), np.std(precs)),
        'rec': (np.mean(recs), np.std(recs)),
        'f1': (np.mean(f1s), np.std(f1s)),
        'loss_history': np.mean(fold_losses, axis=0)
    }

    print(f"{config_name:}: Acc:{metrics['acc'][0]:.4f},{metrics['acc'][1]:.4f} F1:{metrics['f1'][0]:.4f},{metrics['f1'][1]:.4f}")
    return metrics


def naive_baseline():

    accs = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(all_data_filtered,
                                                             labels, groups=groups)) :


        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]


        majority_class = majority_class = 1 if sum(train_labels) > len(train_labels) / 2 else 0
        correct = sum(1 for l in test_labels if l == majority_class)

        acc = correct/len(test_labels)
        accs.append(acc)    

        mean, std = np.mean(accs), np.std(accs)

    print(f"{'Naive baseline (majority class)'}: {mean:.4f}, {std:.4f}")

    return mean, std
