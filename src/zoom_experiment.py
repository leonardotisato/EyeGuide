import os
from typing import Dict, Tuple, Any
import csv
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from utils.seed import set_seeds
from utils.model import ResNet18Classifier
from utils.dataset import FundusClsDataset, FundusClsDatasetZoom, prepare_dataframes
from utils.transforms_512_light import train_transform_class, test_transform_class
from utils.generals import getOutFileName   # la tua versione con Europe/Amsterdam
from utils.generals import progress_bar     # se la metti in un file a parte, altrimenti importa da dove l'hai messa
from utils.visualization import visualize_batch
import hydra
from omegaconf import DictConfig

def evaluate_zoom_levels(cfg, n_splits: int = 5, random_seed: int = 42) -> Dict[Any, Tuple[float, float]]:
    """
    Esegue una k-fold cross validation (di default 5-fold) per diversi livelli di "zoom/dilations"
    e ritorna per ciascun livello (mean_F1, std_F1) sulle validation dei fold.

    Ritorna:
        results[zoom] = (mean_f1, std_f1)
    """
    zoom_levels = [0.0, 0.5, 1.0, 'Original']
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        cudnn.benchmark = True
        print("Using CUDA..")



    exp_name = getOutFileName()
    

    # ----------------- Load full dataset  -----------------
    train_df, val_df, _ = prepare_dataframes(cfg)
    full_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)

    if 'label' not in full_df.columns:
        raise ValueError('CSV must contain a label column')

    labels = full_df['label'].values


    n_epochs = cfg.n_epochs
    batch_size = cfg.batch_size

    results: Dict[Any, Tuple[float, float]] = {}

    # ----------------- iterate over different zoom levels -----------------
    for ext in zoom_levels:
        print(f"\n=== Zoom Level: {ext} ===")
        summary_log = f'{cfg.results_dir}/zoom_evaluation_{exp_name}_{ext}.csv'

        print(f"----- Results will be saved in: {summary_log} -----")
        if not os.path.exists(summary_log):
            with open(summary_log, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([
                    'zoom',
                    'fold',
                    'best_epoch',
                    'best_val_loss',
                    'best_val_acc',
                    'best_val_f1',
                    'best_val_precision',
                    'best_val_recall'
                ])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        fold_f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"\n--- Fold {fold_idx + 1}/{n_splits} (zoom={ext}) ---")

            # --------- prepare dataset for this fold ---------
            if isinstance(ext, str) and ext.lower().startswith('orig'):
                # nessun zoom: dataset "normale"
                full_dataset_train = FundusClsDataset(
                    data_csv=full_df,
                    train=True,
                    transform=train_transform_class
                )
                full_dataset_val = FundusClsDataset(
                    data_csv=full_df,
                    train=False,
                    transform=test_transform_class
                )
            else:
                # zoom/dilated
                full_dataset_train = FundusClsDatasetZoom(
                    csv_file=full_df,
                    transform=train_transform_class,
                    dilation_percentage=float(ext),
                    is_training=True,
                    random_seed=random_seed
                )
                full_dataset_val = FundusClsDatasetZoom(
                    csv_file=full_df,
                    transform=test_transform_class,
                    dilation_percentage=float(ext),
                    is_training=False,
                    random_seed=random_seed
                )

            train_subset = Subset(full_dataset_train, train_idx)
            val_subset = Subset(full_dataset_val, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # visualize a batch
            dataiter = iter(train_loader)
            images, labels_batch = dataiter.next()
            visualize_batch(
                images,
                labels_batch,   # true labels
                labels_batch,   # dummy predicted labels, just to see if everything is ok
                savedir=cfg.results_dir,
                savename=f'zoom_{ext}_fold_{fold_idx+1}_train_batch.png'
            )

            # visualize a batch of validation data
            dataiter = iter(val_loader)
            images, labels_batch = dataiter.next()
            visualize_batch(
                images,
                labels_batch,   # true labels
                labels_batch,   # dummy predicted labels
                savedir=cfg.results_dir,
                savename=f'zoom_{ext}_fold_{fold_idx+1}_val_batch.png'
            )

            # --------- initialize model for this fold ---------
            model = ResNet18Classifier(nr_classes=cfg.nr_classes)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=cfg.teacher_lr, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

            best_val_f1 = -1.0
            best_epoch = -1
            best_val_loss = None
            best_val_acc = None
            best_val_precision = None
            best_val_recall = None
            counter = 0
            # --------- Start training loop  ---------
            for epoch in range(n_epochs):
                print(f"\n[Zoom {ext} | Fold {fold_idx + 1}/{n_splits}] Epoch {epoch + 1}/{n_epochs}")
                # -------------------- Train --------------------
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                all_labels_train = []
                all_preds_train = []

                for batch_idx, (inputs, labels_batch) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels_batch = labels_batch.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()
                    all_labels_train.extend(labels_batch.cpu().numpy())
                    all_preds_train.extend(predicted.cpu().numpy())

                    train_acc_batch = 100.0 * correct / total if total > 0 else 0.0
                    msg = 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        running_loss / (batch_idx + 1),
                        train_acc_batch,
                        correct,
                        total
                    )
                    #progress_bar(batch_idx, len(train_loader), msg)

                avg_train_loss = running_loss / len(train_loader)
                train_accuracy = 100.0 * correct / total if total > 0 else 0.0

                # -------------------- Validation --------------------
                model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                all_labels = []
                all_preds = []

                with torch.no_grad():
                    for batch_idx, (inputs, labels_batch) in enumerate(val_loader):
                        inputs = inputs.to(device)
                        labels_batch = labels_batch.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels_batch)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels_batch.size(0)
                        correct_val += (predicted == labels_batch).sum().item()
                        all_labels.extend(labels_batch.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())

                        val_acc_batch = 100.0 * correct_val / total_val if total_val > 0 else 0.0
                        msg = 'Val Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                            val_loss / (batch_idx + 1),
                            val_acc_batch,
                            correct_val,
                            total_val
                        )
                        #progress_bar(batch_idx, len(val_loader), msg)

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100.0 * correct_val / total_val if total_val > 0 else 0.0

                val_f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
                val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0
                val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0

                print(
                    f"\n[Zoom {ext} | Fold {fold_idx + 1}] "
                    f"Epoch {epoch + 1}/{n_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
                    f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}% | "
                    f"F1: {val_f1:.2f}, Prec: {val_precision:.2f}, Rec: {val_recall:.2f}"
                )

                scheduler.step()

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    best_val_loss = avg_val_loss
                    best_val_acc = val_accuracy
                    best_val_precision = val_precision
                    best_val_recall = val_recall
                    # save model
                    model_path = os.path.join(
                        cfg.models_dir,
                        f'zoom_{ext}_fold_{fold_idx+1}_best_model.pth'
                    )
                    torch.save(model.state_dict(), model_path)
                    print(f"--- Validation loss improved! Checkpoint saved to {model_path}")
                else:
                    counter += 1
                if counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            fold_f1_scores.append(best_val_f1)

            # -------------------- Logging --------------------
            with open(summary_log, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([
                    ext,
                    fold_idx,
                    best_epoch,
                    best_val_loss,
                    best_val_acc,
                    best_val_f1,
                    best_val_precision,
                    best_val_recall
                ])

            print(
                f"\n>>> [Zoom {ext} | Fold {fold_idx + 1}] "
                f"BEST Epoch {best_epoch + 1} | "
                f"Val Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.2f}%, "
                f"F1: {best_val_f1:.2f}, Prec: {best_val_precision:.2f}, Rec: {best_val_recall:.2f}"
            )

        # ----------------- Averaging across folds -----------------
        mean_f1 = float(np.mean(fold_f1_scores))
        std_f1 = float(np.std(fold_f1_scores))
        results[ext] = (mean_f1, std_f1)

        print(f"\n=== Zoom {ext} -> mean F1: {mean_f1:.4f}, std: {std_f1:.4f} ===\n")
    
    return results


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("-------> Starting Zoom Level Experiment <-------")
    set_seeds(cfg.RANDOM_SEED)

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    print(f"\n-------> Initialized paths <-------")
    print(f"-------> Models will be saved in: {cfg.models_dir}")
    print(f"-------> Logs will be saved in: {cfg.logs_dir}")
    print(f"-------> Results will be saved in: {cfg.results_dir}")

    zoom_stats = evaluate_zoom_levels(cfg, n_splits=cfg.n_folds, random_seed=cfg.RANDOM_SEED)

    # select zoom with greater f1 mean
    best_zoom, (best_mean_f1, best_std_f1) = max(
        zoom_stats.items(),
        key=lambda kv: kv[1][0]  
    )

    print(f"\n*** Selected zoom level: {best_zoom} "
          f"(mean F1 = {best_mean_f1:.3f} ± {best_std_f1:.3f}) ***\n")


if __name__ == "__main__":
    main()
