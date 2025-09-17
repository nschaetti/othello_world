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
import sys
import time
import argparse
from typing import Tuple, List, Any, Optional, Dict, Union

# Import rich for better formatting and traceback
from rich import print
from rich.console import Console
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)
console = Console()

# Import ML packages
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import local modules
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.utils import sample, set_seed
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a GPT model on Othello game data')

    # Data root
    parser.add_argument(
        "--data-root",
        type=str,
        help="Root directory of the Othello game data",
    )

    # Synthetic
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Train on synthetic dataset'
    )

    # Number of samples
    parser.add_argument(
        '--num-samples',
        type=int,
        default=-1,
        help='Number of samples to generate (-1 is 20m)'
    )

    # OOD percentage
    parser.add_argument(
        '--ood-perc',
        type=float,
        help='Percent of OOD data to use'
    )
    
    # Model configuration arguments
    parser.add_argument(
        '--n-layer',
        type=int,
        default=8,
        help='Number of transformer layers (default: 8)'
    )

    parser.add_argument(
        '--n-head',
        type=int,
        default=8,
        help='Number of attention heads (default: 8)'
    )

    parser.add_argument(
        '--n-embd',
        type=int,
        default=512,
        help='Embedding dimension (default: 512)'
    )
    
    # Training configuration arguments
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=250,
        help='Maximum number of training epochs (default: 250)'
    )

    # Batch size
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4096,
        help='Batch size for training (default: 4096)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4,
        help='Learning rate (default: 5e-4)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=44,
        help='Random seed for reproducibility (default: 44)'
    )
    
    # Checkpoint arguments
    parser.add_argument('--ckpt-path', type=str, default="./ckpts",
                      help='Path to save the model checkpoint (default: ./ckpts)')
    parser.add_argument('--load_model', action='store_true',
                      help='Load a pre-trained model instead of training a new one')
    parser.add_argument('--load-path', type=str, default=None,
                      help='Path to load the model checkpoint from (default: ./ckpts/gpt_synthetic.ckpt or ./ckpts/gpt_championship.ckpt)')

    # Validation arguments
    parser.add_argument('--validate', action='store_true',
                      help='Validate the model after training')
    parser.add_argument('--val_samples', type=int, default=1000,
                      help='Number of validation samples (default: 1000)')
    
    return parser.parse_args()
# end def parse_arguments


def check_dependencies() -> None:
    """
    Check if all required dependencies are installed.
    
    This function checks if all required packages are available and
    prints a message if any are missing.
    """
    required_packages = ["torch", "numpy", "tqdm"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    # end for
    
    if missing_packages:
        console.print(f"[bold red]Error: Missing required packages: {', '.join(missing_packages)}[/bold red]")
        console.print("Please install them using: [bold]pip install " + " ".join(missing_packages) + "[/bold]")
        sys.exit(1)
    # end if
# end def check_dependencies


def train_model(
        args: argparse.Namespace
) -> Tuple[GPT, CharDataset, torch.device]:
    """
    Train the GPT model on Othello game data.
    
    Args:
        args: Command line arguments
    
    Returns:
        Tuple containing:
            model: Trained GPT model
            train_dataset: Dataset used for training
            device: Device used for training
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Determine data source
    synthetic_or_championship = args.synthetic
    
    # Load Othello data
    console.print("[bold blue]Loading Othello data...[/bold blue]")
    othello = get_othello(
        num_samples=args.num_samples,
        synthetic=args.synthetic,
        data_root=None if synthetic_or_championship else "data/othello-synthetic",
        wthor=True
    )
    # end get_othello
    
    # Create dataset
    console.print("[bold blue]Creating dataset...[/bold blue]")
    train_dataset = CharDataset(othello)
    
    # Configure model
    console.print("[bold blue]Configuring model...[/bold blue]")
    mconf = GPTConfig(
        train_dataset.vocab_size, 
        train_dataset.block_size, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd
    )
    # end GPTConfig
    model = GPT(mconf)
    
    # Generate timestamp for checkpoint path
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    
    # Configure trainer
    console.print("[bold blue]Configuring trainer...[/bold blue]")
    ckpt_path = os.path.join(args.ckpt_path, f"gpt_at{t_start}.ckpt")
    tconf = TrainerConfig(
        max_epochs=args.max_epochs, 
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=True, 
        warmup_tokens=len(train_dataset) * train_dataset.block_size * 5, 
        final_tokens=len(train_dataset) * train_dataset.block_size * args.max_epochs,
        num_workers=0, 
        ckpt_path=ckpt_path
    )
    # end TrainerConfig
    
    # Initialize trainer
    trainer = Trainer(model, train_dataset, None, tconf)
    device = trainer.device
    
    # Print training information
    console.print(f"[bold green]Training GPT model on {'synthetic' if synthetic_or_championship else 'championship'} data[/bold green]")
    console.print(f"[green]Model configuration: {args.n_layer} layers, {args.n_head} heads, {args.n_embd} embedding dim[/green]")
    console.print(f"[green]Training configuration: {args.max_epochs} epochs, {args.batch_size} batch size, {args.learning_rate} learning rate[/green]")
    console.print(f"[green]Checkpoint will be saved to: {tconf.ckpt_path}[/green]")
    console.print(f"[green]Training start time: {t_start}[/green]")
    
    # Train the model
    console.print("[bold yellow]Starting training...[/bold yellow]")
    trainer.train()
    console.print("[bold green]Training complete![/bold green]")
    
    return model, train_dataset, device
# end def train_model


def validate_model(
    model: GPT, 
    train_dataset: CharDataset, 
    device: torch.device, 
    args: argparse.Namespace
) -> None:
    """
    Validate the model by checking if it generates legal moves.
    
    Args:
        model: Trained GPT model
        train_dataset: Dataset used for training
        device: Device to run validation on
        args: Command line arguments
    """
    # Import required modules are already imported at the top level
    from data.othello import OthelloBoardState
    
    console.print("[bold blue]Loading validation data...[/bold blue]")
    # For GPT trained on both datasets, use the validation set of synthetic for validation
    if not args.synthetic:
        othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    else:
        othello = train_dataset.data
    # end if
    
    console.print(f"[bold yellow]Validating model on {args.val_samples} samples...[/bold yellow]")
    total_nodes = 0
    success_nodes = 0
    
    bar = tqdm(othello.val[:args.val_samples], desc="Validating games")
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
            # end try-except
        # end for
        bar.set_description(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes")
    # end for
    
    pass_rate = success_nodes/total_nodes*100
    error_rate = 1 - success_nodes/total_nodes
    
    console.print(f"[bold green]{pass_rate:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes[/bold green]")
    console.print(f"[bold {'green' if error_rate < 0.1 else 'yellow' if error_rate < 0.3 else 'red'}]Error rate: {error_rate:.4f}[/bold]")
# end def validate_model


def load_pretrained_model(
        data_root: str,
        ckpt_path: str,
        synthetic: bool,
        num_samples: int,
        seed: int,
        n_layer: int = 8,
        n_head: int = 8,
        n_embd: int = 512,
        load_path: Optional[str] = None
) -> Tuple[GPT, CharDataset, torch.device]:
    """
    Load a pre-trained model from a checkpoint.
    
    Args:
        data_root (str): Root directory of the data
        ckpt_path (str): Checkpoint directory path
        seed: Random seed for reproducibility
        synthetic (bool): Whether to use synthetic data (True) or championship data (False)
        num_samples (int): Number of samples to use
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        load_path: Specific checkpoint path to load from (optional)
    
    Returns:
        Tuple containing:
            model: Loaded GPT model
            train_dataset: Dataset used for training
            device: Device the model is loaded on
    """
    # Set random seed for reproducibility
    set_seed(seed)

    # Print
    console.print("[bold blue]Loading pre-trained model...[/bold blue]")
    
    # Load Othello data
    console.print("[bold blue]Loading Othello data...[/bold blue]")
    othello = get_othello(
        num_samples=num_samples,
        synthetic=synthetic,
        data_root=data_root,
        wthor=True
    )
    # end get_othello
    
    # Create dataset
    console.print("[bold blue]Creating dataset...[/bold blue]")
    train_dataset = CharDataset(othello)
    
    # Configure model
    console.print("[bold blue]Configuring model...[/bold blue]")
    mconf = GPTConfig(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        n_layer=n_layer, 
        n_head=n_head, 
        n_embd=n_embd
    )

    # end GPTConfig
    model = GPT(mconf)
    
    # Load model weights
    checkpoint_path = load_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ckpt_path, "gpt_synthetic.ckpt" if synthetic else "gpt_championship.ckpt")
    # end if
    
    console.print(f"[bold yellow]Loading model from {checkpoint_path}...[/bold yellow]")
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Move model to a device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    # end if
    
    console.print("[bold green]Model loaded successfully![/bold green]")
    return model, train_dataset, device
# end def load_pretrained_model


def main() -> None:
    """
    Main function to train and optionally validate the model.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Check dependencies before executing the main functionality
    # This allows the help message to be displayed even if dependencies are missing
    check_dependencies()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.ckpt_path, exist_ok=True)
    
    # Either load a pre-trained model or train a new one
    if args.load_model:
        console.print("[bold blue]Loading pre-trained model...[/bold blue]")
        model, train_dataset, device = load_pretrained_model(
            data_root=args.data_root,
            ckpt_path=args.ckpt_path,
            seed=args.seed,
            synthetic=args.synthetic,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            load_path=args.load_path
        )
    else:
        console.print("[bold blue]Training new model...[/bold blue]")
        model, train_dataset, device = train_model(
            args=args
        )
    # end if
    
    # Validate the model if requested
    if args.validate:
        validate_model(model, train_dataset, device, args)
    # end if
# end def main


if __name__ == "__main__":
    main()
# end if
