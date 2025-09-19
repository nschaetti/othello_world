"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """
    Base GPT configuration class that contains parameters common to all GPT versions.
    
    This class defines the core configuration parameters for the GPT model architecture,
    including model dimensions, dropout rates, and other hyperparameters. It serves as
    a container for all configuration options needed to instantiate a GPT model.
    
    Attributes:
        embed_dropout (float): Dropout rate for embeddings. Default is 0.1.
        residual_dropout (float): Dropout rate for residual connections. Default is 0.1.
        attn_dropout (float): Dropout rate for attention weights. Default is 0.1.
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length the model can process.
    """

    embed_dropout = 0.1
    residual_dropout = 0.1
    attn_dropout = 0.1

    def __init__(
            self,
            vocab_size,
            block_size,
            **kwargs
    ):
        """
        Initialize a GPT configuration object.
        
        This constructor sets up the basic configuration for a GPT model, including
        required parameters like vocabulary size and sequence length (block size),
        as well as any additional parameters provided through kwargs.
        
        Args:
            vocab_size (int): Size of the vocabulary (number of possible tokens).
            block_size (int): Maximum sequence length that the model can process.
            **kwargs: Additional configuration parameters that will be set as attributes.
                Common parameters include:
                - n_layer (int): Number of transformer layers.
                - n_head (int): Number of attention heads.
                - n_embd (int): Dimension of embeddings and hidden states.
        """
        self.vocab_size = vocab_size
        self.block_size = block_size

        # Set prop
        for k,v in kwargs.items():
            setattr(self, k, v)
        # end for
    # end __init__

# end class GPTConfig


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    
    This class implements the causal self-attention mechanism, which is a key component
    of transformer models. The "causal" aspect means that each position can only attend
    to previous positions in the sequence, which is essential for autoregressive models
    like GPT. This implementation uses a mask to ensure that attention is only applied
    to positions that come before the current position.
    
    It is possible to use torch.nn.MultiheadAttention here, but this implementation
    provides an explicit version to demonstrate the underlying mechanics.
    
    Attributes:
        key (nn.Linear): Linear projection for creating key vectors.
        query (nn.Linear): Linear projection for creating query vectors.
        value (nn.Linear): Linear projection for creating value vectors.
        attn_drop (nn.Dropout): Dropout layer applied to attention weights.
        resid_drop (nn.Dropout): Dropout layer applied to output projections.
        proj (nn.Linear): Output projection.
        mask (torch.Tensor): Causal mask to ensure attention is only applied to the left.
        n_head (int): Number of attention heads.
    """

    def __init__(
            self,
            config
    ):
        """
        Initialize the causal self-attention layer.
        
        Sets up the key, query, and value projections for all heads, as well as
        the output projection and dropout layers. Also creates a causal mask
        to ensure that attention is only applied to previous positions.
        
        Args:
            config (GPTConfig): Configuration object containing parameters like:
                - n_embd: Embedding dimension
                - n_head: Number of attention heads
                - attn_dropout: Dropout rate for attention weights
                - residual_dropout: Dropout rate for residual connections
                - block_size: Maximum sequence length
        """
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.resid_drop = nn.Dropout(config.residual_dropout)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
    # end __init__

    def forward(self, x, layer_past=None, only_last=-1):
        """
        Forward pass for the causal self-attention layer.
        
        This method computes the self-attention mechanism, where each position in the sequence
        attends to all previous positions. It projects the input into query, key, and value
        vectors, computes attention scores, applies a causal mask, and produces the output.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, embedding_dim].
            layer_past (torch.Tensor, optional): Cached key and value tensors from previous
                forward passes, used for incremental decoding. Defaults to None.
            only_last (int, optional): If > 0, restricts attention to only the last `only_last`
                tokens, preventing earlier tokens from attending to later ones. Defaults to -1
                (no restriction).
                
        Returns:
            tuple:
                - y (torch.Tensor): Output tensor after self-attention and projection,
                  shape [batch_size, seq_length, embedding_dim].
                - att (torch.Tensor): Attention weights, shape [batch_size, n_heads, seq_length, seq_length].
        """
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        if only_last != -1:
            att[:, :, -only_last:, :-only_last] = float('-inf')
        # end if
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    # end forward

# end class CausalSelfAttention

class Block(nn.Module):
    """
    A standard Transformer block.
    
    This class implements a single transformer block, which is a fundamental building
    block of transformer-based architectures. Each block consists of a multi-head
    self-attention layer followed by a feed-forward neural network (MLP), with
    layer normalization and residual connections applied before each sub-layer.
    
    Attributes:
        ln1 (nn.LayerNorm): Layer normalization before the self-attention layer.
        ln2 (nn.LayerNorm): Layer normalization before the MLP.
        attn (CausalSelfAttention): Multi-head causal self-attention layer.
        mlp (nn.Sequential): Feed-forward neural network with GELU activation.
    """

    def __init__(self, config):
        """
        Initialize a Transformer block.
        
        Sets up the layer normalization layers, self-attention mechanism, and
        feed-forward neural network that make up a transformer block.
        
        Args:
            config (GPTConfig): Configuration object containing parameters like:
                - n_embd: Embedding dimension
                - residual_dropout: Dropout rate for residual connections
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.residual_dropout),
        )
    # end __init__

    def forward(self, x, return_att=False, only_last=-1):
        """
        Forward pass for the Transformer block.
        
        This method applies the full transformer block processing sequence:
        1. Layer normalization followed by self-attention
        2. Residual connection from the input
        3. Layer normalization followed by MLP
        4. Residual connection from the intermediate result
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, embedding_dim].
            return_att (bool, optional): Whether to return attention weights along with
                the output. Defaults to False.
            only_last (int, optional): If > 0, restricts attention to only the last `only_last`
                tokens. Passed to the attention layer. Defaults to -1 (no restriction).
                
        Returns:
            torch.Tensor or tuple: 
                - If return_att is False: Output tensor after processing through the block,
                  shape [batch_size, seq_length, embedding_dim].
                - If return_att is True: Tuple of (output tensor, attention weights).
        """
        updt, att = self.attn(self.ln1(x), only_last=only_last)
        x = x + updt
        x = x + self.mlp(self.ln2(x))
        if return_att:
            return x, att
        else:
            return x
        # end if
    # end forward

# end class Block

class GPT(nn.Module):
    """
    The full GPT (Generative Pre-trained Transformer) language model.
    
    This class implements the complete GPT architecture, which consists of:
    1. Token and position embeddings
    2. A stack of transformer blocks
    3. A final layer normalization
    4. A language modeling head (linear layer)
    
    The model processes input token sequences and predicts the next token in the sequence,
    making it suitable for autoregressive text generation and other language modeling tasks.
    
    Attributes:
        token_embedding (nn.Embedding): Embedding layer for input tokens.
        position_embedding (nn.Parameter): Learnable position embeddings.
        drop (nn.Dropout): Dropout layer applied to embeddings.
        blocks (nn.Sequential): Sequence of transformer blocks.
        n_layer (int): Number of transformer layers.
        block_size (int): Maximum sequence length the model can process.
        ln_f (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Output projection to vocabulary size.
    """

    def __init__(
            self,
            config
    ):
        """
        Initialize the GPT model.
        
        Sets up the token and position embeddings, transformer blocks, and output head
        according to the provided configuration.
        
        Args:
            config (GPTConfig): Configuration object containing model parameters like:
                - vocab_size: Size of the vocabulary
                - block_size: Maximum sequence length
                - n_embd: Embedding dimension
                - n_layer: Number of transformer layers
                - embed_dropout: Dropout rate for embeddings
        """
        super().__init__()

        # Embedding layer
        # Token ID -> embedding tensor
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd
        )

        # Position embedding
        # 1, 60 <position>, emb dim
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        # Embedding dropout
        self.drop = nn.Dropout(config.embed_dropout)

        # Transformer, n_layer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # N. layers
        self.n_layer = config.n_layer
        self.block_size = config.block_size

        # Decoder head
        # layer norm
        # linear emb dim -> vocab. size
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Show information
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    # end __init__

    def get_block_size(self):
        """
        Get the maximum sequence length that the model can process.
        
        The block size determines the maximum context length for the model,
        which is the maximum number of tokens that can be processed in a single
        forward pass. This is an important parameter for transformer models as
        it affects the causal attention mask and position embeddings.
        
        Returns:
            int: The maximum sequence length (block size) for this model.
        """
        return self.block_size
    # end def get_block_size

    def _init_weights(self, module):
        """
        Initialize the weights of the model's modules.
        
        This method applies specific initialization strategies to different types of layers:
        - Linear and Embedding layers: Weights are initialized from a normal distribution
          with mean 0.0 and standard deviation 0.02. Biases in Linear layers are set to zero.
        - LayerNorm layers: Biases are set to zero and weights are set to 1.0.
        
        This initialization scheme is commonly used in transformer models and helps
        with training stability and convergence.
        
        Args:
            module (nn.Module): The PyTorch module whose weights will be initialized.
                This can be any layer type in the model.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Init weight with normal
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                # Bias is zero
                module.bias.data.zero_()
            # end if
        elif isinstance(module, nn.LayerNorm):
            # Zero bias, 1.0 for norm.
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # end if
    # end _init_weights

    def configure_optimizers(
            self,
            train_config
    ):
        """
        Configure the optimizer for training the model.
        
        This method sets up the AdamW optimizer with weight decay applied selectively
        to certain parameters. It separates the model parameters into two groups:
        
        1. Parameters that should have weight decay applied (typically weights in linear layers)
        2. Parameters that should not have weight decay applied (biases, layer normalization
           weights, embedding weights, and position embeddings)
        
        This selective application of weight decay is a common practice in transformer models
        and helps improve training stability and model performance.
        
        Args:
            train_config: Configuration object for training that contains:
                - weight_decay (float): The weight decay coefficient to apply
                - learning_rate (float): The learning rate for the optimizer
                - betas (tuple): Beta parameters for the AdamW optimizer
                
        Returns:
            torch.optim.AdamW: Configured optimizer with parameter groups set up for
                selective weight decay.
                
        Raises:
            AssertionError: If any parameter is not properly assigned to a decay/no-decay group
                or if there's overlap between the groups.
        """
        # Separate out all parameters to those that will and
        # won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # For each named module
        # -> name and module
        for mn, m in self.named_modules():
            # -> name, parameter
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                # end if
            # end for
        # end for

        # Special case the position embedding parameter
        # in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Inter & Union
        inter_params = decay & no_decay
        union_params = decay | no_decay

        # Sure there is no param left
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, (
                "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        )

        # Create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

        return optimizer
    # end configure_optimizers

    def forward(
            self,
            idx,
            targets=None
    ):
        """
        Forward pass through the entire GPT model.
        
        This method performs the complete forward pass through the model:
        1. Convert input token indices to embeddings and add position embeddings
        2. Apply dropout to the combined embeddings
        3. Process the embeddings through the transformer blocks
        4. Apply the final layer normalization
        5. Project to vocabulary size to get logits
        6. Calculate loss if targets are provided
        
        Args:
            idx (torch.Tensor): Tensor of token indices of shape [batch_size, sequence_length].
                Each value is an integer representing a token in the vocabulary.
            targets (torch.Tensor, optional): Target token indices of shape [batch_size, sequence_length]
                used for calculating the language modeling loss. If provided, the loss is computed
                and returned along with the logits. Defaults to None.
                
        Returns:
            tuple:
                - logits (torch.Tensor): Output logits of shape [batch_size, sequence_length, vocab_size],
                  representing the predicted probability distribution over the vocabulary for each position.
                - loss (torch.Tensor or None): If targets are provided, this is the cross-entropy loss.
                  Otherwise, it's None.
                  
        Raises:
            AssertionError: If the input sequence length exceeds the model's maximum block size.
        """
        # Both of shape [B, T]
        b, t = idx.size()

        #
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # Forward token embedding
        token_embeddings = self.token_embedding(idx)

        # Get position embedding for the length t
        # 1, 60 <position>, emb dim
        position_embeddings = self.position_embedding[:, :t, :]

        # Add pos. embedding to token embedding => dropout
        x = self.drop(token_embeddings + position_embeddings)

        # Forward blocks
        x = self.blocks(x)

        # Layer norm.
        # [B, T, f]
        x = self.ln_f(x)

        # [B, T, # Words]
        logits = self.head(x)

        # If we are given some desired targets
        # also calculate the loss
        loss = None
        if targets is not None:
            # -100 in the string space is mapped to 0 in the index space
            loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=targets.view(-1),
                ignore_index=0
            )
        # end if

        return logits, loss
    # end forward

# end class GPT


class GPTforProbing(GPT):
    """
    A variant of the GPT model designed for probing internal representations.
    
    This class extends the base GPT model to allow access to the internal activations
    at a specified layer. This is useful for analyzing and understanding the model's
    internal representations, conducting interpretability research, or extracting
    features for downstream tasks.
    
    Attributes:
        probe_layer (int): The index of the layer after which activations will be probed.
            If -1, uses the final layer.
        ln (bool): Whether to apply layer normalization to the probed activations.
    """
    
    def __init__(
            self,
            config,
            probe_layer=-1,
            ln=False
    ):
        """
        Initialize the GPT model for probing.
        
        Sets up a GPT model that can be used to extract internal representations
        from a specific layer.
        
        Args:
            config (GPTConfig): Configuration object for the GPT model.
            probe_layer (int, optional): The index of the layer after which to probe
                activations. If -1, uses the final layer. Defaults to -1.
            ln (bool, optional): Whether to apply layer normalization to the probed
                activations. Defaults to False.
                
        Raises:
            AssertionError: If the specified probe_layer is not a valid layer index.
        """
        super(GPTforProbing, self).__init__(config)

        # We probe the activation after the self.probe_layer-th layer
        self.probe_layer = self.n_layer if probe_layer == -1 else probe_layer

        # Check probe layer validity
        assert self.probe_layer <= self.n_layer and self.probe_layer >= 0, "Invalid layer index to probe"

        self.ln = ln
    # end __init__
        
    def forward(self, idx, return_att=False):
        """
        Forward pass that returns internal activations from a specified layer.
        
        This method processes the input through the model up to the specified probe_layer
        and returns the activations at that point. Optionally, it can also return
        attention weights from the last processed layer.
        
        Args:
            idx (torch.Tensor): Tensor of token indices of shape [batch_size, sequence_length].
                Each value is an integer representing a token in the vocabulary.
            return_att (bool, optional): Whether to return attention weights from the
                last processed layer. Defaults to False.
                
        Returns:
            torch.Tensor or tuple:
                - If return_att is False: Activations tensor after processing through
                  the specified number of layers, shape [batch_size, sequence_length, embedding_dim].
                - If return_att is True: Tuple of (activations tensor, attention weights).
                
        Raises:
            AssertionError: If the input sequence length exceeds the model's maximum block size.
        """
        b, t = idx.size()  # both of shape [B, T]

        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # Token embedding
        # each index maps to a (learnable) vector
        token_embeddings = self.token_embedding(idx)

        # Position embeddings
        # each position maps to a (learnable) vector
        position_embeddings = self.position_embedding[:, :t, :] # each position maps to a (learnable) vector

        # Dropout
        x = self.drop(token_embeddings + position_embeddings)

        # Forward through block
        att = None
        for b in self.blocks[:self.probe_layer]:
            if return_att:
                x, att = b(x, return_att=return_att)
            else:
                x = b(x, return_att=return_att)
            # end if
        # end for
        
        # Layer norm
        if self.ln:
            x = self.ln_f(x)  # [B, T, f]
        # end if

        # Return output or (+) attention
        if return_att:
            return x, att
        else:
            return x
        # end if
    # end forward

# end class GPTforProbing


class GPTforIntervention(GPT):
    """
    A variant of the GPT model designed for interventional analysis.
    
    This class extends the base GPT model to allow for interventions at specific layers.
    It splits the forward pass into two stages, allowing for modifications to the
    intermediate representations between these stages. This is particularly useful
    for causal analysis, interpretability research, and studying how changes to
    internal representations affect model outputs.
    
    Attributes:
        probe_layer (int): The index of the layer at which the model will be split
            for intervention. If -1, uses the final layer.
    """
    
    def __init__(
            self,
            config,
            probe_layer: int = -1
    ):
        """
        Initialize the GPT model for interventional analysis.
        
        Sets up a GPT model that can be used to perform interventions at a specific layer.
        
        Args:
            config (GPTConfig): Configuration object for the GPT model.
            probe_layer (int, optional): The index of the layer at which to split the model
                for intervention. If -1, uses the final layer. Defaults to -1.
                
        Raises:
            AssertionError: If the specified probe_layer is not a valid layer index.
                Must be between 1 and n_layer inclusive.
        """
        super(GPTforIntervention, self).__init__(config)

        # we probe the activation after the self.probe_layer-th layer 
        self.probe_layer = self.n_layer if probe_layer == -1 else probe_layer

        assert self.probe_layer <= self.n_layer and self.probe_layer >= 1, "Invalid layer index to probe"
    # end __init__
        
    def forward_1st_stage(self, idx):
        """
        First stage of the forward pass, up to the intervention point.
        
        This method processes the input through the model up to the specified probe_layer,
        returning the intermediate activations at that point. These activations can then
        be modified before being passed to the second stage.
        
        Args:
            idx (torch.Tensor): Tensor of token indices of shape [batch_size, sequence_length].
                Each value is an integer representing a token in the vocabulary.
                
        Returns:
            torch.Tensor: Intermediate activations after processing through the first
                probe_layer layers, shape [batch_size, sequence_length, embedding_dim].
                
        Raises:
            AssertionError: If the input sequence length exceeds the model's maximum block size.
        """
        b, t = idx.size()  # both of shape [B, T]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.token_embedding(idx) # each index maps to a (learnable) vector
        position_embeddings = self.position_embedding[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        
        for b in self.blocks[:self.probe_layer]:
            x = b(x)
        # end for
        
        # x = self.blocks(x)
        # x = self.ln_f(x)  # [B, T, f]
        # logits = self.head(x)  # [B, T, # Words]
        return x
    # end forward_1st_stage
    
    def forward_2nd_stage(self, x, targets=None, only_last=-1):
        """
        Second stage of the forward pass, from the intervention point to the output.
        
        This method continues processing from the intermediate activations (which may have
        been modified after the first stage) through the remaining layers of the model,
        and optionally calculates the loss if targets are provided.
        
        Args:
            x (torch.Tensor): Intermediate activations from the first stage or modified
                activations, shape [batch_size, sequence_length, embedding_dim].
            targets (torch.Tensor, optional): Target token indices of shape 
                [batch_size, sequence_length] used for calculating the language modeling loss.
                If provided, the loss is computed and returned along with the logits.
                Defaults to None.
            only_last (int, optional): If > 0, restricts attention to only the last `only_last`
                tokens in the remaining layers. Defaults to -1 (no restriction).
                
        Returns:
            tuple:
                - logits (torch.Tensor): Output logits of shape [batch_size, sequence_length, vocab_size],
                  representing the predicted probability distribution over the vocabulary for each position.
                - loss (torch.Tensor or None): If targets are provided, this is the cross-entropy loss.
                  Otherwise, it's None.
        """        
        for b in self.blocks[self.probe_layer:]:
            x = b(x, only_last=only_last)
        # end for
        x = self.ln_f(x)  # [B, T, f]
        logits = self.head(x)  # [B, T, # Words]
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        # end if
        return logits, loss
    # end forward_2nd_stage

# end class GPTforIntervention

class GPTforProbeIA(GPT):
    """
    A variant of the GPT model designed for probing interactions between layers.
    
    This class extends the base GPT model to analyze how representations at different
    layers interact with each other. It allows for processing inputs through specific
    ranges of layers and collecting intermediate activations, which is useful for
    studying how information flows through the model and how different layers
    contribute to the final prediction.
    
    Attributes:
        probe_layer (int): The index of the initial layer from which to start probing.
            If -1, uses the final layer.
    """
    
    def __init__(self, config, probe_layer=-1):
        """
        Initialize the GPT model for probing layer interactions.
        
        Sets up a GPT model that can be used to analyze interactions between
        representations at different layers.
        
        Args:
            config (GPTConfig): Configuration object for the GPT model.
            probe_layer (int, optional): The index of the initial layer from which
                to start probing. If -1, uses the final layer. Defaults to -1.
                
        Raises:
            AssertionError: If the specified probe_layer is not a valid layer index.
                Must be between 0 and n_layer inclusive.
        """
        super(GPTforProbeIA, self).__init__(config)
        # we probe the activation after the self.probe_layer-th layer 
        self.probe_layer = self.n_layer if probe_layer == -1 else probe_layer
        assert self.probe_layer <= self.n_layer and self.probe_layer >= 0, "Invalid layer index to probe"
    # end __init__
        
    def forward_1st_stage(self, idx):
        """
        First stage of the forward pass, up to the initial probe layer.
        
        This method processes the input through the model up to the specified probe_layer,
        returning the intermediate activations at that point. These activations can then
        be passed to the second stage for further processing through specific layer ranges.
        
        Args:
            idx (torch.Tensor): Tensor of token indices of shape [batch_size, sequence_length].
                Each value is an integer representing a token in the vocabulary.
                
        Returns:
            torch.Tensor: Intermediate activations after processing through the first
                probe_layer layers, shape [batch_size, sequence_length, embedding_dim].
                
        Raises:
            AssertionError: If the input sequence length exceeds the model's maximum block size.
        """
        b, t = idx.size()  # both of shape [B, T]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.token_embedding(idx) # each index maps to a (learnable) vector
        position_embeddings = self.position_embedding[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        
        for b in self.blocks[:self.probe_layer]:
            x = b(x)
        # end for
        
        # x = self.blocks(x)
        # x = self.ln_f(x)  # [B, T, f]
        # logits = self.head(x)  # [B, T, # Words]
        return x
    # end forward_1st_stage
    
    def forward_2nd_stage(self, x, start_layer, end_layer=-1):
        """
        Second stage of the forward pass, processing through a specific range of layers.
        
        This method continues processing from the intermediate activations through a
        specified range of layers, collecting the activations at each layer. This allows
        for analyzing how representations evolve through different parts of the model.
        
        Args:
            x (torch.Tensor): Intermediate activations from the first stage,
                shape [batch_size, sequence_length, embedding_dim].
            start_layer (int): The index of the first layer to process in this stage.
            end_layer (int, optional): The index of the layer at which to stop processing
                (exclusive). If -1, processes through all remaining layers. Defaults to -1.
                
        Returns:
            list: A list of activation tensors, one for each layer processed, each with
                shape [batch_size, sequence_length, embedding_dim].
        """
        tbr = []
        if end_layer == -1:
            end_layer = self.n_layer + 1
        # end if
        for b in self.blocks[start_layer: end_layer]:
            x = b(x)
            tbr.append(x)
        # end for
        # x = self.ln_f(x)  # [B, T, f]
        return tbr
    # end forward_2nd_stage
    
    def predict(self, x, targets=None):
        """
        Generate predictions from intermediate activations.
        
        This method takes intermediate activations (typically from one of the previous stages)
        and processes them through the final layer normalization and output head to generate
        logits. It can also calculate the loss if targets are provided.
        
        Args:
            x (torch.Tensor): Intermediate activations, shape [batch_size, sequence_length, embedding_dim].
            targets (torch.Tensor, optional): Target token indices of shape 
                [batch_size, sequence_length] used for calculating the language modeling loss.
                If provided, the loss is computed and returned along with the logits.
                Defaults to None.
                
        Returns:
            tuple:
                - logits (torch.Tensor): Output logits of shape [batch_size, sequence_length, vocab_size],
                  representing the predicted probability distribution over the vocabulary for each position.
                - loss (torch.Tensor or None): If targets are provided, this is the cross-entropy loss.
                  Otherwise, it's None.
        """
        x = self.ln_f(x)  # [B, T, f]
        logits = self.head(x)  # [B, T, # Words]
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        # end if
        return logits, loss
    # end predict

# end class GPTforProbeIA