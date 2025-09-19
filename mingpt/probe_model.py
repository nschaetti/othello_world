"""
Probe models for classification tasks.

This module contains probe models used for classification tasks, particularly
designed for analyzing neural network representations in the context of games like Othello.
The probes are designed to perform multiple classification tasks simultaneously.
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

class BatteryProbeClassification(nn.Module):
    """
    A single-layer probe model for multiple classification tasks.
    
    This class implements a linear probe that can be used for multiple classification tasks
    simultaneously. It's particularly designed for analyzing neural network representations
    in the context of games like Othello, where each position on the board can be treated
    as a separate classification task.
    
    Attributes:
        input_dim (int): Dimension of the input features.
        probe_class (int): Number of classes for each classification task.
        num_task (int): Number of classification tasks.
        proj (nn.Linear): Linear projection layer for classification.
    """
    def __init__(self, device, probe_class, num_task, input_dim=512):  # from 0 to 15
        """
        Initialize the battery probe classification model.
        
        Args:
            device (torch.device): Device to place the model on (CPU or GPU).
            probe_class (int): Number of classes for each classification task.
            num_task (int): Number of classification tasks.
            input_dim (int, optional): Dimension of the input features. Default is 512.
        """
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        self.num_task = num_task
        self.proj = nn.Linear(self.input_dim, self.probe_class * self.num_task, bias=True)
        self.apply(self._init_weights)
        self.to(device)
    
    def forward(self, act, y=None):
        """
        Forward pass of the battery probe classification model.
        
        This method processes input activations and optionally computes the loss
        if target labels are provided.
        
        Args:
            act (torch.Tensor): Input activations with shape [batch_size, input_dim].
            y (torch.Tensor, optional): Target labels with shape [batch_size, num_task].
                If None, only logits are returned. Default is None.
                
        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Predicted logits with shape [batch_size, num_task, probe_class].
                - loss (torch.Tensor or None): Cross entropy loss if targets are provided, None otherwise.
        """
        # [B, f], [B, #task]
        logits = self.proj(act).reshape(-1, self.num_task, self.probe_class)  # [B, #task, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        
        This method is applied to each module in the model during initialization.
        It initializes the weights of linear and embedding layers with a normal distribution,
        and sets biases to zero. For LayerNorm layers, it sets biases to zero and weights to one.
        
        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def configure_optimizers(self, train_config):
        """
        Configure optimizers for training the model.
        
        This method separates model parameters into two groups: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        It then creates and returns a PyTorch optimizer and learning rate scheduler.
        
        Args:
            train_config: Configuration object containing training parameters like:
                - weight_decay: Weight decay factor for regularization.
                - learning_rate: Learning rate for the optimizer.
                - betas: Beta parameters for the Adam optimizer.
                
        Returns:
            tuple: A tuple containing:
                - optimizer (torch.optim.Optimizer): The configured optimizer.
                - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler
    
class BatteryProbeClassificationTwoLayer(nn.Module):
    """
    A two-layer probe model for multiple classification tasks.
    
    This class implements a two-layer neural network probe that can be used for multiple 
    classification tasks simultaneously. It's particularly designed for analyzing neural 
    network representations in the context of games like Othello, where each position on 
    the board can be treated as a separate classification task. The two-layer architecture 
    allows for more complex mappings between input features and classification outputs.
    
    Attributes:
        input_dim (int): Dimension of the input features.
        probe_class (int): Number of classes for each classification task.
        num_task (int): Number of classification tasks.
        mid_dim (int): Dimension of the hidden layer.
        proj (nn.Sequential): Sequential module containing the two-layer network.
    """
    def __init__(self, device, probe_class, num_task, mid_dim, input_dim=512):  # from 0 to 15
        """
        Initialize the two-layer battery probe classification model.
        
        Args:
            device (torch.device): Device to place the model on (CPU or GPU).
            probe_class (int): Number of classes for each classification task.
            num_task (int): Number of classification tasks.
            mid_dim (int): Dimension of the hidden layer.
            input_dim (int, optional): Dimension of the input features. Default is 512.
        """
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        self.num_task = num_task
        self.mid_dim = mid_dim
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.mid_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.mid_dim, self.probe_class * self.num_task, bias=True),
        )
        self.apply(self._init_weights)
        self.to(device)
    
    def forward(self, act, y=None):
        """
        Forward pass of the two-layer battery probe classification model.
        
        This method processes input activations through the two-layer network and 
        optionally computes the loss if target labels are provided.
        
        Args:
            act (torch.Tensor): Input activations with shape [batch_size, input_dim].
            y (torch.Tensor, optional): Target labels with shape [batch_size, num_task].
                If None, only logits are returned. Default is None.
                
        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Predicted logits with shape [batch_size, num_task, probe_class].
                - loss (torch.Tensor or None): Cross entropy loss if targets are provided, None otherwise.
        """
        # [B, f], [B, #task]
        logits = self.proj(act).reshape(-1, self.num_task, self.probe_class)  # [B, #task, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        
        This method is applied to each module in the model during initialization.
        It initializes the weights of linear and embedding layers with a normal distribution,
        and sets biases to zero. For LayerNorm layers, it sets biases to zero and weights to one.
        
        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def configure_optimizers(self, train_config):
        """
        Configure optimizers for training the model.
        
        This method separates model parameters into two groups: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        It then creates and returns a PyTorch optimizer and learning rate scheduler.
        
        Args:
            train_config: Configuration object containing training parameters like:
                - weight_decay: Weight decay factor for regularization.
                - learning_rate: Learning rate for the optimizer.
                - betas: Beta parameters for the Adam optimizer.
                
        Returns:
            tuple: A tuple containing:
                - optimizer (torch.optim.Optimizer): The configured optimizer.
                - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler