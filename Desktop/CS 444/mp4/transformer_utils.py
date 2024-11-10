import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Compute self attention by scaled dot product. 
    ``query``, ``key``, and ``value`` are computed from input token features
    using linear layers. Similarity is computed using Scaled Dot-Product
    Attention where the dot product is scaled by a factor of square root of the
    dimension of the query vectors. See ``Attention Is All You Need" for more details.

    Args for __init__:
        input_dim (int): input dimention of attention
        query_dim (int): query dimention of attention
        key_dim (int): key dimention of attention
        value_dim (int): value dimention of attention

    Inputs for forward function: 
        x (batch, num_tokens, input_dim): batch of input feature vectors for the tokens.
    Outputs from forward function:
        attn_output (batch, num_tokens, value_dim): outputs after self-attention
    """

    def __init__(self, input_dim, query_dim, key_dim, value_dim):
        super(SelfAttention, self).__init__()
        assert(query_dim == key_dim)
        self.query_dim = query_dim
        self.input_dim = input_dim
        

        self.W_query = nn.Linear(input_dim, query_dim)
        self.W_key = nn.Linear(input_dim, key_dim)
        self.W_value = nn.Linear(input_dim, value_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        # TODO(student)
        # Do not use the attention implementation in pytorch!!
        
        # 1. Use W_query, W_key, W_value to compute query, key and value representations from the input token features
        
        # 2. compute similarity by dot product Query with Key.T and divided by a scale

        # 3. pass them softmax to make sure similarities are between [0, 1] range

        # 4. multiply with value

         # Step 1: Compute queries, keys, and values
        queries = self.W_query(x)  # Shape: (batch_size, num_tokens, query_dim)
        keys = self.W_key(x)       # Shape: (batch_size, num_tokens, key_dim)
        values = self.W_value(x)   # Shape: (batch_size, num_tokens, value_dim)

        # Step 2: Compute scaled dot-product attention scores
        # Transpose keys for batch matrix multiplication
        # keys.transpose(1, 2): (batch_size, key_dim, num_tokens)
        # attention_scores: (batch_size, num_tokens, num_tokens)
        scale = torch.sqrt(torch.tensor(self.query_dim, dtype=queries.dtype, device=queries.device))
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / scale

        # Step 3: Apply softmax to get attention weights
        # attention_weights: (batch_size, num_tokens, num_tokens)
        attention_weights = self.softmax(attention_scores)

        # Step 4: Multiply attention weights with values
        # attn_output: (batch_size, num_tokens, value_dim)
        attn_output = torch.bmm(attention_weights, values)

        return attn_output


class LayerNorm(nn.Module):
    """
    Args: input_dim, epsilon
        input_dim (int): dimensionality of input feature vectors
        epsilon (float): epsilon for when normalizing by the variance.

    Input to forward function:
        x (batch, num_tokens, input_dim): input features for tokens.

    Output from forward function:
        x_out (batch, num_tokens, input_dim): token features after layer normalization.
    """

    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        assert isinstance(input_dim, int)

        self.input_dim = input_dim
        self.eps = eps
        
        # w: the learnable weights initialized to 1.
        self.w = nn.Parameter(torch.ones(self.input_dim))
        
        # b: the learnable bias initialized to 0.
        self.b = nn.Parameter(torch.zeros(self.input_dim))
    
    def forward(self, x: torch.Tensor):
        assert(x.shape[-1] == self.input_dim)
        # TODO (student) 

        # input: (batch_size: N, seq_length: C, hidden_dim: D)
        # 1. calculate the mean of all elements (make sure you're taking the mean and variation over the d_model dimension)

        # 2. calculate the variance of all element

        # 3. calculate normalized x

        # 4. apply scale and shift(the w and b parameters)

        # Step 1: Calculate the mean over the input_dim (C) dimension
        mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch_size, seq_length, 1)

        # Step 2: Calculate the variance over the input_dim (C) dimension
        variance = x.var(dim=-1, keepdim=True, unbiased=True)  # Shape: (batch_size, seq_length, 1)

        # Step 3: Normalize the input
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)  # Shape: (batch_size, seq_length, input_dim)

        # Step 4: Apply the learnable scale (w) and shift (b)
        x_out = x_normalized * self.w + self.b  # Broadcasting over batch and seq_length dimensions

        return x_out
