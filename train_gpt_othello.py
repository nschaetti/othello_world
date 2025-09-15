#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Othello-GPT Model

This script trains a GPT model on Othello game data and saves the model to a checkpoint file.
The model can be trained on either synthetic data or championship data.

The script is based on the minGPT implementation by Andrej Karpathy.

Usage:
    python train_gpt_othello.py [--synthetic] [--championship] [--max_epochs MAX_EPOCHS] 
                               [--batch_size BATCH_SIZE] [--learning_rate LR]
                               [--n_layer N_LAYER] [--n_head N_HEAD] [--n_embd N_EMBD]
                               [--ckpt_path CKPT_PATH] [--validate]

Requirements:
    - Python 3.6+
    - PyTorch 1.0+
    - numpy
    - tqdm
    - pgn (for loading PGN files)
    - matplotlib (for visualization)
    - seaborn (for visualization)

You can install the required packages using pip:
    pip install torch numpy tqdm pgn matplotlib seaborn

Author: Original notebook author
Date: 2025-09-14
"""

import os
import math
import time
import argparse
import sys
import importlib
from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F

# Import basic packages needed for argument parsing
import numpy as np
import torch

from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.utils import sample, set_seed
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig


# Check for required dependencies
def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'numpy', 'tqdm', 'pgn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Error: The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using pip:")
        print(f"  pip install {' '.join(missing_packages)}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a GPT model on Othello game data')
    
    # Data source arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--synthetic', action='store_true', 
                          help='Train on synthetic dataset')
    data_group.add_argument('--championship', action='store_true',
                          help='Train on championship dataset')
    
    # Model configuration arguments
    parser.add_argument('--n_layer', type=int, default=8,
                      help='Number of transformer layers (default: 8)')
    parser.add_argument('--n_head', type=int, default=8,
                      help='Number of attention heads (default: 8)')
    parser.add_argument('--n_embd', type=int, default=512,
                      help='Embedding dimension (default: 512)')
    
    # Training configuration arguments
    parser.add_argument('--max_epochs', type=int, default=250,
                      help='Maximum number of training epochs (default: 250)')
    parser.add_argument('--batch-size', type=int, default=4096,
                      help='Batch size for training (default: 4096)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                      help='Learning rate (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=44,
                      help='Random seed for reproducibility (default: 44)')
    
    # Checkpoint arguments
    parser.add_argument('--ckpt_path', type=str, default=None,
                      help='Path to save the model checkpoint (default: ./ckpts/gpt_<timestamp>.ckpt)')
    parser.add_argument('--load_model', action='store_true',
                      help='Load a pre-trained model instead of training a new one')
    parser.add_argument('--load_path', type=str, default=None,
                      help='Path to load the model checkpoint from (default: ./ckpts/gpt_synthetic.ckpt or ./ckpts/gpt_championship.ckpt)')
    
    # Validation arguments
    parser.add_argument('--validate', action='store_true',
                      help='Validate the model after training')
    parser.add_argument('--val_samples', type=int, default=1000,
                      help='Number of validation samples (default: 1000)')
    
    return parser.parse_args()


def train_model(args):
    """
    Train the GPT model on Othello game data.
    
    Args:
        args: Command line arguments
    
    Returns:
        model: Trained GPT model
        train_dataset: Dataset used for training
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Determine data source
    synthetic_or_championship = args.synthetic
    
    # Load Othello data
    print("Loading Othello data...")
    othello = get_othello(
        ood_num=-1, 
        data_root=None if synthetic_or_championship else "data/othello-synthetic",
        wthor=True
    )
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = CharDataset(othello)
    
    # Configure model
    print("Configuring model...")
    mconf = GPTConfig(
        train_dataset.vocab_size, 
        train_dataset.block_size, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd
    )
    model = GPT(mconf)
    
    # Generate timestamp for checkpoint path
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    
    # Configure trainer
    print("Configuring trainer...")
    tconf = TrainerConfig(
        max_epochs=args.max_epochs, 
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=True, 
        warmup_tokens=len(train_dataset) * train_dataset.block_size * 5, 
        final_tokens=len(train_dataset) * train_dataset.block_size * args.max_epochs,
        num_workers=0, 
        ckpt_path=args.ckpt_path if args.ckpt_path else f"./ckpts/gpt_at{t_start}.ckpt", 
    )
    
    # Initialize trainer
    trainer = Trainer(model, train_dataset, None, tconf)
    device = trainer.device
    
    # Print training information
    print(f"Training GPT model on {'synthetic' if synthetic_or_championship else 'championship'} data")
    print(f"Model configuration: {args.n_layer} layers, {args.n_head} heads, {args.n_embd} embedding dim")
    print(f"Training configuration: {args.max_epochs} epochs, {args.batch_size} batch size, {args.learning_rate} learning rate")
    print(f"Checkpoint will be saved to: {tconf.ckpt_path}")
    print(f"Training start time: {t_start}")
    
    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training complete!")
    
    return model, train_dataset, device


def validate_model(model, train_dataset, device, args):
    """
    Validate the model by checking if it generates legal moves.
    
    Args:
        model: Trained GPT model
        train_dataset: Dataset used for training
        device: Device to run validation on
        args: Command line arguments
    """
    # Import required modules
    from tqdm import tqdm
    from data import get_othello
    from data.othello import OthelloBoardState
    from mingpt.utils import sample
    
    print("Loading validation data...")
    # For GPT trained on both datasets, use the validation set of synthetic for validation
    if not args.synthetic:
        othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    else:
        othello = train_dataset.data
    
    print(f"Validating model on {args.val_samples} samples...")
    total_nodes = 0
    success_nodes = 0
    
    bar = tqdm(othello.val[:args.val_samples])
    for whole_game in bar:
        length_of_whole_game = len(whole_game)
        for length_of_partial_game in range(1, length_of_whole_game):
            total_nodes += 1
            context = whole_game[:length_of_partial_game]
            x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(device)
            y = sample(model, x, 1, temperature=1.0)[0]
            completion = [train_dataset.itos[int(i)] for i in y if i != -1]
            try:
                OthelloBoardState().update(completion, prt=False)
            except Exception:
                pass
            else:
                success_nodes += 1
        bar.set_description(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes")
    
    print(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes")
    print(f"Error rate: {1 - success_nodes/total_nodes:.4f}")


def load_pretrained_model(args):
    """
    Load a pre-trained model from a checkpoint.
    
    Args:
        args: Command line arguments
    
    Returns:
        model: Loaded GPT model
        train_dataset: Dataset used for training
        device: Device the model is loaded on
    """
    # Import required modules
    from data import get_othello
    from mingpt.dataset import CharDataset
    from mingpt.utils import set_seed
    from mingpt.model import GPT, GPTConfig
    
    print("Loading pre-trained model...")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Determine data source
    synthetic_or_championship = args.synthetic
    
    # Load Othello data
    print("Loading Othello data...")
    othello = get_othello(
        ood_num=-1, 
        data_root=None if synthetic_or_championship else "data/othello-synthetic",
        wthor=True
    )
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = CharDataset(othello)
    
    # Configure model
    print("Configuring model...")
    mconf = GPTConfig(
        train_dataset.vocab_size, 
        train_dataset.block_size, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd
    )
    model = GPT(mconf)
    
    # Load model weights
    checkpoint_path = args.load_path
    if checkpoint_path is None:
        checkpoint_path = "./ckpts/gpt_synthetic.ckpt" if args.synthetic else "./ckpts/gpt_championship.ckpt"
    
    print(f"Loading model from {checkpoint_path}...")
    load_res = model.load_state_dict(torch.load(checkpoint_path))
    
    # Move model to device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    
    print("Model loaded successfully!")
    return model, train_dataset, device


def main():
    """
    Main function to train and optionally validate the model.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Check dependencies before executing main functionality
    # This allows the help message to be displayed even if dependencies are missing
    check_dependencies()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs("./ckpts", exist_ok=True)
    
    # Either load a pre-trained model or train a new one
    if args.load_model:
        model, train_dataset, device = load_pretrained_model(args)
    else:
        model, train_dataset, device = train_model(args)
    # end if
    
    # Validate the model if requested
    if args.validate:
        validate_model(model, train_dataset, device, args)
    # end if
# end def main


if __name__ == "__main__":
    main()
# end if
