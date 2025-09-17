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
from typing import Tuple, List, Dict, Any, Optional, Union

# Import rich for better formatting and traceback
from rich import print
from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler

# Install rich traceback handler
install(show_locals=True)
console = Console()

# Set up logging with rich
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

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
    def __init__(self, act: List[torch.Tensor], y: List[int], age: List[List[int]]) -> None:
        """
        Initialize the ProbingDataset.
        
        Args:
            act: List of activations from the model
            y: List of target properties to predict
            age: List of game state ages
        """
        assert len(act) == len(y), "Activations and targets must have the same length"
        assert len(act) == len(age), "Activations and ages must have the same length"
        
        console.print(f"[bold blue]{len(act)} pairs loaded...[/bold blue]")
        self.act = act
        self.y = y
        self.age = age
        
        # Print class distribution
        class_0 = np.sum(np.array(y)==0)
        class_1 = np.sum(np.array(y)==1)
        class_2 = np.sum(np.array(y)==2)
        console.print(f"[green]Class distribution: Class 0: {class_0}, Class 1: {class_1}, Class 2: {class_2}[/green]")
        
        # Count age distribution
        long_age = []
        for a in age:
            long_age.extend(a)
        # end for
        
        long_age = np.array(long_age)
        counts = [np.count_nonzero(long_age == i) for i in range(60)]
        del long_age
        console.print(f"[green]Age distribution: {counts}[/green]")
    # end def __init__
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.y)
    # end def __len__
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing:
                torch.Tensor: Activation
                torch.Tensor: Target property
                torch.Tensor: Age
        """
        return (
            self.act[idx],
            torch.tensor(self.y[idx]).to(torch.long),
            torch.tensor(self.age[idx]).to(torch.long)
        )
    # end def __getitem__
# end class ProbingDataset


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following attributes:
            layer (int): Layer to extract representations from
            epo (int): Number of training epochs
            mid_dim (int): Middle dimension for two-layer probe
            twolayer (bool): Whether to use two-layer probe
            random (bool): Whether to use randomly initialized model
            championship (bool): Whether to use model trained on championship data
            exp (str): Experiment type
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

    # Non linear probe?
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


def create_folder_name(args: argparse.Namespace) -> str:
    """
    Create folder name for saving results based on configuration.
    
    Args:
        args: Command line arguments with experiment configuration
        
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
    # end if
    
    if args.championship:
        folder_name = folder_name + "_championship"
    # end if
    
    console.print(f"[bold blue]Running experiment for [/bold blue][bold green]{folder_name}[/bold green]")
    return folder_name
# end def create_folder_name


def load_dataset() -> CharDataset:
    """
    Load Othello dataset from the championship dataset.
    
    Returns:
        CharDataset: Loaded dataset containing Othello game data
    """
    console.print("[bold blue]Loading Othello dataset...[/bold blue]")
    othello = get_othello(data_root="data/championship-dataset")
    return CharDataset(othello)
# end def load_dataset


def initialize_model(args: argparse.Namespace, train_dataset: CharDataset) -> GPTforProbing:
    """
    Initialize and configure the GPT model.
    
    Args:
        args: Command line arguments containing model configuration
        train_dataset: Training dataset with vocabulary and block size information
        
    Returns:
        GPTforProbing: Initialized model configured for probing
    """
    console.print("[bold blue]Initializing GPT model...[/bold blue]")
    
    # Initialize GPT model configuration
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPTforProbing(mconf, probe_layer=args.layer)
    
    # Load model weights based on configuration
    if args.random:
        # Use randomly initialized weights
        console.print("[yellow]Using randomly initialized weights[/yellow]")
        model.apply(model._init_weights)
    elif args.championship:
        # Load model trained on championship data
        console.print("[yellow]Loading model trained on championship data[/yellow]")
        model.load_state_dict(torch.load("./checkpoints/gpt_championship.ckpt"))
    else:
        # Load model trained on synthetic dataset (default)
        console.print("[yellow]Loading model trained on synthetic dataset[/yellow]")
        model.load_state_dict(torch.load("./checkpoints/gpt_synthetic.ckpt"))
    # end if
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        console.print(f"[green]Moving model to GPU: {device}[/green]")
        model = model.to(device)
    else:
        console.print("[yellow]No GPU available, using CPU[/yellow]")
    # end if
    
    return model
# end def initialize_model


def extract_activations(
    model: GPTforProbing, 
    train_dataset: CharDataset, 
    args: argparse.Namespace
) -> Tuple[List[torch.Tensor], List[int], List[List[int]]]:
    """
    Extract activations from the model for each game state.
    
    Args:
        model: GPT model configured for probing
        train_dataset: Training dataset containing Othello games
        args: Command line arguments with experiment configuration
        
    Returns:
        Tuple containing:
            List[torch.Tensor]: Activations for each game state
            List[int]: Properties for each game state
            List[List[int]]: Ages for each game state
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    console.print(f"[bold blue]Extracting activations using device: [/bold blue][green]{device}[/green]")
    
    # Create data loader
    loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=1)
    
    # Containers for activations and properties
    act_container = []
    property_container = []
    
    # Extract activations and properties
    console.print("[bold blue]Extracting activations and properties...[/bold blue]")
    for x, y in tqdm(loader, total=len(loader), desc="Processing games for activations"):
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
    # end for
    
    # Extract age information for each game state
    console.print("[bold blue]Extracting age information...[/bold blue]")
    age_container = []
    for x, y in tqdm(loader, total=len(loader), desc="Processing games for ages"):
        tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
        valid_until = tbf.index(-100) if -100 in tbf else 999
        
        a = OthelloBoardState()
        ages = a.get_gt(tbf[:valid_until], "get_age")  # [block_size, ]
        age_container.extend(ages)
    # end for
    
    console.print(f"[bold green]Extracted {len(act_container)} activations[/bold green]")
    return act_container, property_container, age_container
# end def extract_activations


def initialize_probe(
    args: argparse.Namespace, 
    device: Union[str, torch.device]
) -> Union[BatteryProbeClassification, BatteryProbeClassificationTwoLayer]:
    """
    Initialize probe model based on configuration.
    
    Args:
        args: Command line arguments with probe configuration
        device: Device to run the model on (CPU or GPU)
        
    Returns:
        Union[BatteryProbeClassification, BatteryProbeClassificationTwoLayer]: 
            Initialized probe model based on configuration
    """
    console.print("[bold blue]Initializing probe model...[/bold blue]")
    
    # Set number of classes based on experiment type
    if args.exp == "state":
        probe_class = 3
    else:
        probe_class = 3  # Default to 3 classes if experiment type is not recognized
    # end if
    
    # Initialize probe model based on configuration
    if args.twolayer:
        console.print(f"[yellow]Creating two-layer probe with middle dimension {args.mid_dim}[/yellow]")
        probe = BatteryProbeClassificationTwoLayer(
            device, 
            probe_class=probe_class, 
            num_task=64, 
            mid_dim=args.mid_dim
        )
    else:
        console.print("[yellow]Creating single-layer probe[/yellow]")
        probe = BatteryProbeClassification(
            device, 
            probe_class=probe_class, 
            num_task=64
        )
    # end if
    
    return probe
# end def initialize_probe


def create_data_loaders(
    act_container: List[torch.Tensor], 
    property_container: List[int], 
    age_container: List[List[int]]
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, DataLoader, DataLoader]:
    """
    Create data loaders for training and testing.
    
    Args:
        act_container: Container for model activations
        property_container: Container for target properties
        age_container: Container for game state ages
        
    Returns:
        Tuple containing:
            torch.utils.data.Subset: Training dataset subset
            torch.utils.data.Subset: Testing dataset subset
            DataLoader: Training data loader
            DataLoader: Testing data loader
    """
    console.print("[bold blue]Creating data loaders...[/bold blue]")
    
    # Create probing dataset
    probing_dataset = ProbingDataset(act_container, property_container, age_container)
    
    # Split into train/test
    train_size = int(0.8 * len(probing_dataset))
    test_size = len(probing_dataset) - train_size
    console.print(f"[green]Splitting dataset: {train_size} training samples, {test_size} testing samples[/green]")
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        probing_dataset, 
        [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        shuffle=False, 
        pin_memory=True, 
        batch_size=128, 
        num_workers=1
    )
    
    test_loader = DataLoader(
        test_dataset, 
        shuffle=True, 
        pin_memory=True, 
        batch_size=128, 
        num_workers=1
    )
    
    console.print("[bold green]Data loaders created successfully[/bold green]")
    return train_dataset, test_dataset, train_loader, test_loader
# end def create_data_loaders


def configure_trainer(
    args: argparse.Namespace, 
    train_dataset: torch.utils.data.Subset, 
    folder_name: str
) -> TrainerConfig:
    """
    Configure trainer for training the probe.
    
    Args:
        args: Command line arguments with training configuration
        train_dataset: Training dataset subset
        folder_name: Folder name for saving results
        
    Returns:
        TrainerConfig: Configuration object for the trainer
    """
    console.print("[bold blue]Configuring trainer...[/bold blue]")
    
    max_epochs = args.epo
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    ckpt_path = os.path.join("./ckpts/", folder_name, f"layer{args.layer}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    # Configure trainer
    trainer_config = TrainerConfig(
        max_epochs=max_epochs, 
        batch_size=1024, 
        learning_rate=1e-3,
        betas=(.9, .999), 
        lr_decay=True, 
        warmup_tokens=len(train_dataset)*5, 
        final_tokens=len(train_dataset)*max_epochs,
        num_workers=4, 
        weight_decay=0., 
        ckpt_path=ckpt_path
    )
    
    console.print(f"[green]Trainer configured with {max_epochs} epochs[/green]")
    console.print(f"[green]Checkpoints will be saved to: {ckpt_path}[/green]")
    
    return trainer_config
# end def configure_trainer


def train_probe(
    probe: Union[BatteryProbeClassification, BatteryProbeClassificationTwoLayer],
    train_dataset: torch.utils.data.Subset,
    test_dataset: torch.utils.data.Subset,
    trainer_config: TrainerConfig
) -> Trainer:
    """
    Train the probe model.
    
    Args:
        probe: Probe model (single-layer or two-layer)
        train_dataset: Training dataset subset
        test_dataset: Testing dataset subset
        trainer_config: Configuration for the trainer
        
    Returns:
        Trainer: Trained trainer object with training history
    """
    console.print("[bold blue]Training probe model...[/bold blue]")
    
    # Initialize trainer
    trainer = Trainer(probe, train_dataset, test_dataset, trainer_config)
    
    # Train the model
    console.print("[bold yellow]Starting training...[/bold yellow]")
    trainer.train(prt=True)
    
    # Save training traces and checkpoint
    console.print("[bold blue]Saving training traces and checkpoint...[/bold blue]")
    trainer.save_traces()
    trainer.save_checkpoint()
    
    console.print("[bold green]Training completed successfully![/bold green]")
    return trainer
# end def train_probe


def main() -> None:
    """
    Main function to orchestrate the training process.
    
    This function coordinates the entire training pipeline:
    1. Parse command line arguments
    2. Create folder name for saving results
    3. Load Othello dataset
    4. Initialize GPT model
    5. Extract activations from the model
    6. Initialize probe model
    7. Create data loaders
    8. Configure trainer
    9. Train probe
    """
    console.print("[bold blue]Starting probe training process[/bold blue]")
    
    # Make deterministic
    set_seed(42)
    console.print("[green]Random seed set to 42 for reproducibility[/green]")
    
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
    console.print(f"[green]Using device: {device}[/green]")
    
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
    
    console.print("[bold green]Probe training process completed successfully![/bold green]")
# end def main


if __name__ == "__main__":
    main()
# end if

