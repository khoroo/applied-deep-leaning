#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple, Dict
import pickle
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from os import path

import pprint
import argparse
from pathlib import Path

from model import fetch_model
from dataset import UrbanSound8KDataset


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a CNN on UrbanSounds-8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--weight-decay", default=1e-5, type=float, help="Weight decay on Adam")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of sounds within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--resume-checkpoint",
    type=Path,
    help='Sets the path (if any) to load existing model',
)
parser.add_argument(
    "--checkpoint-frequency",
    default=10,
    type=int,
    help="Save a checkpoint every N epochs",
)
parser.add_argument(
    "--checkpoint-path",
    default=Path("checkpoint.pkl"),
    type=Path,
    help='Sets the path to save the model',
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--mode",
    default='LMC',
    type=str,
    choices=['LMC', 'MC', 'MLMC'],
    help="What dataset to train on",
)
parser.add_argument(
    "--TSCNN",
    nargs = 2,
    type=Path,
    help="Paths of previously trained LMC and MC models, in order\
          \n Usage: --TSCNN ./LMCmodel.pth ./MCmodel.pth",
)
parser.add_argument(
    "--spec-augment",
    default=0,
    type=float,
    help="Probability of augmenting data, default 0",
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    sound_names = ('ac', 'ch', 'cp', 'db', 'dr', 'ei', 'gs', 'jh', 'si', 'sm')
    if args.TSCNN is not None:
        test_LMC = fetch_loader(False, 'LMC', False)
        test_MC = fetch_loader(False, 'MC', False)
        
        LMCNet = fetch_model('LMC').to(DEVICE)
        MCNet = fetch_model('MC').to(DEVICE)
        
        LMCNet.load_state_dict(torch.load(args.TSCNN[0])['model'])
        MCNet.load_state_dict(torch.load(args.TSCNN[1])['model'])

        logits_LMC, idx_LMC = fetch_TSCNN_file_logits(LMCNet, test_LMC, DEVICE)
        logits_MC, idx_MC = fetch_TSCNN_file_logits(MCNet, test_MC, DEVICE)

        results = {"preds": [], "labels": []}
        apply_softmax = nn.Softmax(dim=-1)
        file_logits = []
        fname_to_file_logits_idx = dict()
        
        for _, labels, fnames in test_LMC:
            labels = labels.to(DEVICE)
            for i, fname in enumerate(fnames):
                logits_sum = apply_softmax(logits_LMC[idx_LMC[fname]]) +\
                             apply_softmax(logits_MC[idx_MC[fname]])
                
                if fname in fname_to_file_logits_idx.keys():
                    # add segment logits to file index
                    file_logits[fname_to_file_logits_idx[fname]] += logits_sum
                else:
                    fname_to_file_logits_idx[fname] = len(file_logits) # get new index
                    file_logits.append(logits_sum)
                    results["labels"].append(labels.cpu().numpy()[i])
        file_logits = torch.stack(file_logits)
        preds = file_logits.argmax(dim=-1).cpu().numpy()
        results["preds"].extend(list(preds))

        results["preds"] = np.array(results["preds"])
        results["labels"] = np.array(results["labels"])
        
        accuracy = compute_accuracy(results["labels"], results["preds"])

        class_accuracy = compute_per_class_accuracy(
            results["labels"], results["preds"], sound_names
        )

        print(f"accuracy: {accuracy * 100:2.2f}")
        print('Per class accuracy:')
        pprint.pprint(class_accuracy)


    else:
        train_loader = fetch_loader(True, args.mode, args.spec_augment)
        test_loader = fetch_loader(False, args.mode, False)
        model = fetch_model(args.mode)

        #LOADING IN THE MODEL IF PATH GIVEN
        if args.resume_checkpoint is not None:
            state_dict = torch.load(args.resume_checkpoint)
            print(f"Loading model from {args.resume_checkpoint}")
            pprint.pprint(state_dict['model'].keys())
            model.load_state_dict(state_dict['model'])

        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        

        log_dir = get_summary_writer_log_dir(args)
        print(f"Writing logs to {log_dir}")
        summary_writer = SummaryWriter(
                str(log_dir),
                flush_secs=5
        )
        trainer = Trainer(
            model, train_loader, test_loader, criterion,
            optimizer, summary_writer, sound_names, DEVICE
        )

        trainer.train(
            args.epochs,
            args.val_frequency,
            print_frequency=args.print_frequency,
            log_frequency=args.log_frequency,
            checkpoint_frequency=args.checkpoint_frequency,
            checkpoint_path=args.checkpoint_path,
        )

        summary_writer.close()
    

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        sound_names: tuple,
        device: torch.device,

    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.sound_names = sound_names
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
        checkpoint_frequency: int = 10,
        checkpoint_path: str = './model.pth'
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels, fnames in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            # save model
            if ((epoch + 1) % checkpoint_frequency) == 0:
                print(f"Saving model to {checkpoint_path}")
                torch.save({
                            'model': self.model.state_dict(),
                            }, checkpoint_path)
            # validate model
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        apply_softmax = nn.Softmax(dim=-1)
        file_logits = []
        fname_to_file_logits_idx = dict()
        
        self.model.eval()     
        
        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels, fnames in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = apply_softmax(self.model(batch))
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                for i, fname in enumerate(fnames):
                    if fname in fname_to_file_logits_idx.keys():
                        # add segment logits to file index
                        file_logits[fname_to_file_logits_idx[fname]] += logits[i]
                    else:
                        fname_to_file_logits_idx[fname] = len(file_logits) # get new index
                        file_logits.append(logits[i])
                        results["labels"].append(labels.cpu().numpy()[i])
            file_logits = torch.stack(file_logits)
            preds = file_logits.argmax(dim=-1).cpu().numpy()
            results["preds"].extend(list(preds))
        
        results["preds"] = np.array(results["preds"])
        results["labels"] = np.array(results["labels"])
        
        accuracy = compute_accuracy(
            results["labels"], results["preds"]
        )

        class_accuracy = compute_per_class_accuracy(
            results["labels"], results["preds"], self.sound_names
        )

        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        print('Per class accuracy:')
        pprint.pprint(class_accuracy)

def fetch_TSCNN_file_logits(model, val_loader, device):
    results = {"preds": [], "labels": []}
    apply_softmax = nn.Softmax(dim=-1)
    file_logits = []
    fname_to_file_logits_idx = dict()
    
    model.eval()     
    
    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, labels, fnames in val_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            logits = apply_softmax(model(batch))
            for i, fname in enumerate(fnames):
                if fname in fname_to_file_logits_idx.keys():
                    # add segment logits to file index
                    file_logits[fname_to_file_logits_idx[fname]] += logits[i]
                else:
                    fname_to_file_logits_idx[fname] = len(file_logits) # get new index
                    file_logits.append(logits[i])
    return file_logits, fname_to_file_logits_idx

def compute_per_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
    label_names: tuple
) -> Dict[str, float]:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
        label_name: tuple of string labels for classifier
    """
    
    class_accuracy = {}
    for i, label_name in enumerate(label_names):
        class_accuracy[label_name] = np.sum((labels==preds) & (labels==i)) / np.sum((labels==i))
    return class_accuracy

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
      f"{args.mode}-Net_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      f"weight_decay={args.weight_decay}_"
      f"spec-aug-{args.spec_augment}_"
      f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def fetch_loader(
    train: bool,
    mode: str,
    spec_augment: float,
    freq_mask=20, # shape of logmelspec
    time_mask=5,
    ):
    """
    train: is loader train True/False (else test)
    mode: what data type to load
    spec_augment: augment data or not
    """
    if train:
        path = 'UrbanSound8K_train.pkl'
    else:
        path = 'UrbanSound8K_test.pkl'

    loader =  torch.utils.data.DataLoader(
        UrbanSound8KDataset(path, mode, spec_augment, freq_mask, time_mask),
        batch_size=32, 
        shuffle=train,
        num_workers=8, 
        pin_memory=True
    )
    return loader


    

if __name__ == "__main__":
    main(parser.parse_args())