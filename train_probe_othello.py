#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Probe for Othello GPT Model

This script trains a probe classifier on representations from a GPT model trained on Othello game data.
The probe is designed to predict certain properties of the Othello game state from the model's internal
representations at a specified layer.

Author: Original author
Date: Original date
"""

# Standard library imports
import os
import time
import argparse

# Set up logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

# Local imports
from mingpt.utils import set_seed
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbing
from mingpt.probe_trainer import Trainer, TrainerConfig
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer


class ProbingDataset(Dataset):
    """
    Dataset class for training and evaluating probes.
    
    Attributes:
        act: List of activations from the model
        y: List of target properties to predict
        age: List of game state ages
    """
    def __init__(self, act, y, age):
        assert len(act) == len(y)
        assert len(act) == len(age)
        print(f"{len(act)} pairs loaded...")
        self.act = act
        self.y = y
        self.age = age
        
        # Print class distribution
        print(np.sum(np.array(y)==0), np.sum(np.array(y)==1), np.sum(np.array(y)==2))
        
        # Count age distribution
        long_age = []
        for a in age:
            long_age.extend(a)
        # end for
        
        long_age = np.array(long_age)
        counts = [np.count_nonzero(long_age == i) for i in range(60)]
        del long_age
        print(counts)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.act[idx],
            torch.tensor(self.y[idx]).to(torch.long),
            torch.tensor(self.age[idx]).to(torch.long)
        )
    # end def __getitem__

# end class ProbingDataset


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train classification network')

    # Which layer used for training
    parser.add_argument(
        '--layer',
        required=True,
        default=-1,
        type=int,
        help='Layer to extract representations from'
    )

    # How many epochs
    parser.add_argument(
        '--epo',
        default=16,
        type=int,
        help='Number of training epochs'
    )

    # Hidden layer dimension
    parser.add_argument(
        '--mid-dim',
        default=128,
        type=int,
        help='Middle dimension for two-layer probe'
    )

    # Non linear probe ?
    parser.add_argument(
        '--twolayer',
        dest='twolayer',
        action='store_true',
        help='Use two-layer probe instead of single layer'
    )

    # Random baseline
    parser.add_argument(
        '--random',
        dest='random',
        action='store_true',
        help='Use randomly initialized model'
    )

    # Use model trained on championship data
    parser.add_argument(
        '--championship',
        dest='championship',
        action='store_true',
        help='Use model trained on championship data'
    )

    # Experiment type
    parser.add_argument(
        '--exp',
        default="state",
        type=str,
        help='Experiment type (e.g., state)'
    )
    
    return parser.parse_known_args()[0]
# end def parse_arguments


def create_folder_name(args):
    """
    Create folder name for saving results based on configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Folder name for saving results
    """
    folder_name = f"battery_othello/{args.exp}"
    
    # Add suffixes based on configuration
    if args.twolayer:
        folder_name = folder_name + f"_tl{args.mid_dim}"  # tl for probes without batchnorm
    # end if
    
    if args.random:
        folder_name = folder_name + "_random"
    
    if args.championship:
        folder_name = folder_name + "_championship"
    
    print(f"Running experiment for {folder_name}")
    return folder_name


def load_dataset():
    """
    Load Othello dataset.
    
    Returns:
        CharDataset: Loaded dataset
    """
    othello = get_othello(data_root="data/championship-dataset")
    return CharDataset(othello)
# end load_dataset


def initialize_model(args, train_dataset):
    """
    Initialize and configure the GPT model.
    
    Args:
        args: Command line arguments
        train_dataset: Training dataset
        
    Returns:
        GPTforProbing: Initialized model
    """
    # Initialize GPT model configuration
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPTforProbing(mconf, probe_layer=args.layer)
    
    # Load model weights based on configuration
    if args.random:
        # Use randomly initialized weights
        model.apply(model._init_weights)
    elif args.championship:
        # Load model trained on championship data
        model.load_state_dict(torch.load("./checkpoints/gpt_championship.ckpt"))
    else:
        # Load model trained on synthetic dataset (default)
        model.load_state_dict(torch.load("./checkpoints/gpt_synthetic.ckpt"))
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    
    return model


def extract_activations(model, train_dataset, args):
    """
    Extract activations from the model for each game state.
    
    Args:
        model: GPT model
        train_dataset: Training dataset
        args: Command line arguments
        
    Returns:
        tuple: Tuple containing activations, properties, and ages
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    # Create data loader
    loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=1)
    
    # Containers for activations and properties
    act_container = []
    property_container = []
    
    # Extract activations and properties
    for x, y in tqdm(loader, total=len(loader)):
        # Convert token indices to characters
        tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
        valid_until = tbf.index(-100) if -100 in tbf else 999
        
        # Create Othello board state and get properties
        a = OthelloBoardState()
        properties = a.get_gt(tbf[:valid_until], "get_" + args.exp)  # [block_size, ]
        
        # Get model activations
        act = model(x.to(device))[0, ...].detach().cpu()  # [block_size, f]
        
        # Store activations and properties
        act_container.extend([_[0] for _ in act.split(1, dim=0)[:valid_until]])
        property_container.extend(properties)
    
    # Extract age information for each game state
    age_container = []
    for x, y in tqdm(loader, total=len(loader)):
        tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
        valid_until = tbf.index(-100) if -100 in tbf else 999
        
        a = OthelloBoardState()
        ages = a.get_gt(tbf[:valid_until], "get_age")  # [block_size, ]
        age_container.extend(ages)
    
    return act_container, property_container, age_container


def initialize_probe(args, device):
    """
    Initialize probe model based on configuration.
    
    Args:
        args: Command line arguments
        device: Device to run the model on
        
    Returns:
        BatteryProbeClassification or BatteryProbeClassificationTwoLayer: Initialized probe model
    """
    # Set number of classes based on experiment type
    if args.exp == "state":
        probe_class = 3
    
    # Initialize probe model based on configuration
    if args.twolayer:
        probe = BatteryProbeClassificationTwoLayer(device, probe_class=probe_class, num_task=64, mid_dim=args.mid_dim)
    else:
        probe = BatteryProbeClassification(device, probe_class=probe_class, num_task=64)
    
    return probe


def create_data_loaders(act_container, property_container, age_container):
    """
    Create data loaders for training and testing.
    
    Args:
        act_container: Container for activations
        property_container: Container for properties
        age_container: Container for ages
        
    Returns:
        tuple: Tuple containing train_dataset, test_dataset, train_loader, test_loader
    """
    # Create probing dataset
    probing_dataset = ProbingDataset(act_container, property_container, age_container)
    
    # Split into train/test
    train_size = int(0.8 * len(probing_dataset))
    test_size = len(probing_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(probing_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=128, num_workers=1)
    test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=128, num_workers=1)
    
    return train_dataset, test_dataset, train_loader, test_loader


def configure_trainer(args, train_dataset, folder_name):
    """
    Configure trainer for training the probe.
    
    Args:
        args: Command line arguments
        train_dataset: Training dataset
        folder_name: Folder name for saving results
        
    Returns:
        TrainerConfig: Trainer configuration
    """
    max_epochs = args.epo
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    
    return TrainerConfig(
        max_epochs=max_epochs, 
        batch_size=1024, 
        learning_rate=1e-3,
        betas=(.9, .999), 
        lr_decay=True, 
        warmup_tokens=len(train_dataset)*5, 
        final_tokens=len(train_dataset)*max_epochs,
        num_workers=4, 
        weight_decay=0., 
        ckpt_path=os.path.join("./ckpts/", folder_name, f"layer{args.layer}")
    )


def train_probe(probe, train_dataset, test_dataset, trainer_config):
    """
    Train the probe model.
    
    Args:
        probe: Probe model
        train_dataset: Training dataset
        test_dataset: Testing dataset
        trainer_config: Trainer configuration
        
    Returns:
        Trainer: Trained trainer
    """
    trainer = Trainer(probe, train_dataset, test_dataset, trainer_config)
    trainer.train(prt=True)
    trainer.save_traces()
    trainer.save_checkpoint()
    
    return trainer


def main():
    """
    Main function to orchestrate the training process.
    """
    # Make deterministic
    set_seed(42)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create folder name for saving results
    folder_name = create_folder_name(args)
    
    # Load Othello dataset
    train_dataset = load_dataset()
    
    # Initialize GPT model
    model = initialize_model(args, train_dataset)
    
    # Extract activations from the model
    act_container, property_container, age_container = extract_activations(model, train_dataset, args)
    
    # Get device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    # Initialize probe model
    probe = initialize_probe(args, device)
    
    # Create data loaders
    train_dataset, test_dataset, train_loader, test_loader = create_data_loaders(
        act_container, property_container, age_container
    )
    
    # Configure trainer
    trainer_config = configure_trainer(args, train_dataset, folder_name)
    
    # Train probe
    trainer = train_probe(probe, train_dataset, test_dataset, trainer_config)
# end def main


if __name__ == "__main__":
    main()
# end if

