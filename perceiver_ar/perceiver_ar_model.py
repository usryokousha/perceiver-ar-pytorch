"""Perceiver AR architecture and components 
transcribed from DeepMind Jax Version."""

from concurrent.futures import process
import functools
import math
from optparse import Option
from typing import Any, Callable, List, Optional, Sequence, Tuple

from dataclasses import dataclass
from functorch import vmap

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import weight_init

import numpy as np


@dataclass
class Masks:
    encoder: torch.Tensor
    processor: torch.Tensor


@dataclass
class Output:
    input_events_logits: torch.Tensor
    encoder_mask: torch.Tensor
    processor_mask: torch.Tensor
    latent_last_steps: torch.Tensor
    perceiver_state: Sequence[torch.Tensor]


def get_sequence_length(sequence):
    """Return the length of non-zero entries in the sequence."""
    # Return the first index where a 0 occurs.
    length = torch.argmax(sequence == 0)

    # If argmax returns 0, that means that either
    # 1) No 0s were found, and the sequence length is the full length of the array
    # 2) There's padding immediately at the beginning, indicating that the array
    #    is all padding and the sequence length is 0.
    length = torch.where(torch.logical_and(length == 0, sequence[0] != 0),
                         sequence.shape[0], length)

    return length

def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, torch.stack(ys)

def truncate_sequence(sequence):
    """Replace final token in sequence with padding."""
    length = get_sequence_length(sequence)
    sequence[torch.maximum(0, length - 1)] = 0
    return sequence


def fill_diagonal(a, val):
    """Fill the diagonal of the last two dimensions of an array with a value."""
    assert a.ndim >= 2
    a.fill_diagonal_(val, dim1=-2, dim2=-1)
    return a


def make_positions_terminal_relative(pos_seq, input_seq):
    """Convert positions or position encodings to terminal-relative coords."""
    # From [0, 1, 2, ..., T, x, x, ..., x] to:
    #                                     [T, ..., 2, 1, 0, x, x, ..., x]
    #                                            last input ^
    seq_len = get_sequence_length(input_seq)
    pos_seq = torch.flip(pos_seq)
    return torch.roll(pos_seq, seq_len, dims=0)


def conv_1d(
        input_channels,
        output_channels,
        init_scale=1.0,
        with_bias=True):
    """A 1D convolution."""
    conv = nn.Linear(
        in_features=input_channels,
        out_features=output_channels,
        bias=with_bias)
    weight_init.variance_scaling_(conv.weight.data, scale=init_scale)
    weight_init.variance_scaling_(conv.bias.data, scale=init_scale)
    return conv


def make_attention_mask(query_input: torch.Tensor,
                        key_input: torch.Tensor,
                        pairwise_fn: Callable[..., Any] = torch.mul,
                        extra_batch_dims: int = 0,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
    attention weights will be `[batch..., heads, len_q, len_kv]` and this
    function will produce `[batch..., 1, len_q, len_kv]`.

    Args:
        query_input: a batched, flat input of query_length size
        key_input: a batched, flat input of key_length size
        pairwise_fn: broadcasting elementwise comparison function
        extra_batch_dims: number of extra batch dims to add singleton
        axes for, none by default
        dtype: mask return dtype

    Returns:
        A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
    mask.unsqueeze_(-3)
    mask = mask.view(mask, *([1] * extra_batch_dims), -1)
    return mask.to(dtype)


def combine_masks(*masks: Optional[torch.Tensor],
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Combine attention masks.

    Args:
      *masks: set of attention mask arguments to combine, some can be None.
      dtype: dtype for the returned mask.

    Returns:
      Combined mask, reduced by logical and, returns None if no masks given.
    """
    masks_list = [m for m in masks if m is not None]
    if not masks_list:
        return None
    assert all(map(lambda x: x.ndim == masks_list[0].ndim, masks_list)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks_list))}')
    mask, *other_masks = masks_list
    for other_mask in other_masks:
        mask = torch.logical_and(mask, other_mask)
    return mask.to(dtype)


def generate_sinusoidal_features(size,
                                 max_len=2048,
                                 min_scale=1.0,
                                 max_scale=10000.0,
                                 device=torch.device('cpu')):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
        size: embedding size.
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.

    Returns:
        output: init function returning `(max_len, size)`
    """

    pe = torch.zeros((max_len, size), dtype=torch.float32, device=device)
    position = torch.arange(0, max_len)[:, None]
    scale_factor = -torch.log(max_scale / min_scale) / (size // 2 - 1)
    div_term = min_scale * torch.exp(torch.arange(0, size // 2) * scale_factor)
    pe[:, :size // 2] = torch.sin(position * div_term)
    pe[:, size // 2: 2 * (size // 2)] = torch.cos(position * div_term)
    return pe


def generate_linear_features(size,
                             max_len=2048):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
      size: embedding size.
      max_len: maximum possible length for the input.

    Returns:
      output: init function returning `(max_len, size)`
    """
    position = np.arange(0, max_len)[:, None]
    return torch.broadcast_to(position, (max_len, size)).to(torch.float32)


def generate_fourier_features(pos, n_bands, max_res=224, concat_pos=True):
    """Generate a Fourier frequency position encoding with linear spacing.

    Args:
        pos: The position of n points in d dimensional space.
        A jnp array of shape [n, d].
        n_bands: The number of bands (K) to use.
        max_res: The maximum resolution (i.e. the number of pixels per dim).
        concat_pos: Concatenate the input position encoding to the Fourier features?
    Returns:
        embedding: A 1D jnp array of shape [n, d * (1 + 2 * n_bands)].
        Output dimensions are ordered as
        [dim_1, dim_2, ..., dim_d,
        sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
        sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
        cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
        cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
        where dim_i is pos[:, i] and f_k is the kth frequency band.
    """
    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    max_freq = max_res / 2
    freq_bands = torch.linspace(min_freq, max_freq, steps=n_bands)

    # Get frequency bands for each spatial dimension.
    pos_freq_bands = torch.einsum('nd, k->ndk', pos, freq_bands)
    pos_freq_bands = torch.reshape(pos_freq_bands,
                                   [-1, np.prod(pos_freq_bands.shape[1:])])

    # Output is size [n, 2 * d * n_bands]
    encoding = torch.concat(
        [torch.sin(np.pi * pos_freq_bands),
         torch.cos(np.pi * pos_freq_bands)], dim=-1)
    # Concatenate the raw input positions.
    if concat_pos:
        encoding = torch.concat([pos, encoding], dim=-1)
    return encoding


def build_linear_positions(index_dims, output_range=(-1.0, 1.0), device=torch.device('cpu')):
    """Generate an array of position indices for an N-D input array.

    Args:
        index_dims: The shape of the index dimensions of the input array.
        output_range: The min and max values taken by each input index dimension.
    Returns:
        A torch tensor of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
    """
    def _linspace(n_xels_per_dim):
        return torch.linspace(
            output_range[0], output_range[1],
            steps=n_xels_per_dim,
            dtype=torch.float32,
            device=device)

    dim_ranges = [
        _linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges, indexing='ij')

    return torch.stack(array_index_grid, dim=-1)


def bernoulli(p: float = 0.5,
              shape: Optional[Sequence[int]] = None):
    return torch.empty(shape).uniform_(0, 1).bernoulli_(p)


def make_block_causal_masks(
        inputs,
        latent_index_dim: int,
        latents_per_position: int,
        batch_size: int,
        mask_style: str,
        latent_dropout_prob: float,
        is_training: bool):
    """Constructs block-causal attention masks."""
    # Latents divide the sequence as evenly as possible.
    input_index_dim = inputs.shape[1]

    def batch_broadcast(arr):
        return torch.broadcast_to(arr, (batch_size,) + arr.shape)

    input_id = torch.arange(start=0, stop=input_index_dim, dtype=torch.float32)
    input_id = batch_broadcast(input_id)

    if mask_style == 'final_block':
        # Place all latents at the end with a stride of 1. Only one latent is
        # placed at each trailing position past the first, invalid ones are
        # discarded.
        assert latent_index_dim % latents_per_position == 0

        def get_steps(events):
            sequence_length = get_sequence_length(events)
            num_unique_positions = latent_index_dim // latents_per_position
            last_steps = sequence_length * torch.ones(
                [num_unique_positions], dtype=torch.int32)
            offsets = torch.arange(start=-num_unique_positions, stop=0, step=1)
            last_steps += offsets
            last_steps = torch.maximum(last_steps, -1)  # -1 means invalid
            return last_steps
    else:
        raise ValueError(f'Unknown mask_style: {mask_style}.')

    # latent_last_steps is B x latent_index_dim // latents_per_position x C
    # It's used for position attention to avoid duplicated computation.
    # all_latent_positions is B x latent_index_dim x C
    # It's used to construct masks (except the loss mask), because it shows
    # how latents are related to each other.
    latent_last_steps = get_steps(inputs)
    all_latent_positions = latent_last_steps.repeat(1, latents_per_position)

    encoder_mask_raw = make_attention_mask(
        query_input=all_latent_positions, key_input=input_id,
        pairwise_fn=torch.greater_equal, dtype=inputs.dtype)

    # Mask invalid inputs as well:
    input_mask = inputs > 0
    input_mask_array = make_attention_mask(
        query_input=torch.ones(
            [batch_size, latent_index_dim], dtype=inputs.dtype, device=inputs.device),
        key_input=input_mask)
    encoder_mask_final = encoder_mask_raw * input_mask_array

    # Latents can pool from latents with the same or earlier final inputs.
    processor_mask = make_attention_mask(
        query_input=all_latent_positions, key_input=all_latent_positions,
        pairwise_fn=torch.greater_equal, dtype=inputs.dtype)

    # Mask any invalid latents (i.e. with negative index):
    key_is_valid = (all_latent_positions >= 0).astype(
        all_latent_positions.dtype)
    valid_latent_mask = make_attention_mask(
        query_input=torch.ones(
            [batch_size, latent_index_dim], device=inputs.device),
        key_input=key_is_valid)
    processor_mask = combine_masks(
        processor_mask, valid_latent_mask)

    if is_training:
        # Drop out latents: they can't influence any other latents.
        keep_rate = 1.0 - latent_dropout_prob
        latent_dropout_keys = bernoulli(
            keep_rate, shape=[batch_size, latent_index_dim])
        latent_dropout_mask = make_attention_mask(
            query_input=torch.ones(
                [batch_size, latent_index_dim], device=inputs.device),
            key_input=latent_dropout_keys)
        processor_mask = combine_masks(
            processor_mask, latent_dropout_mask)

    # Force diagonal to be unmasked.
    processor_mask = fill_diagonal(processor_mask, 1.0)

    masks = Masks(encoder=encoder_mask_final, processor=processor_mask)

    return masks, latent_last_steps


def _make_rotation_matrices(
    x: torch.Tensor,
    max_wavelength: int,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds the cosine and sine matrices used to compute rotary embeddings.

    Args:
      x: The array the rotary embeddings will be applied to.
      max_wavelength: Maximum wavelength that will appear in sin/cosine waveforms.
        This specifies the maximum sequence length for identifying unique
        positions.
      positions: A [B, T] tensor of positions.
    Returns:
      cos_matrix: [B, 1, T, head_dim] cosine component of the embedding rotation.
      sin_matrix: [B, 1, T, head_dim] sine component of the embedding rotation.
    """
    batch_size, seq_len, _, head_dim = x.shape

    # head_dim is assumed to be constructed/padded so it's even
    assert head_dim % 2 == 0

    # Generated log-spaced wavelengths between 1 and the max_wavelength.
    num_bands = head_dim // 2
    freq = max_wavelength**((2./head_dim) *
                            torch.linspace(0, num_bands, num_bands,
                            dtype=positions.dtype, device=positions.device))
    inv_freq = 1. / freq
    inv_freq = torch.repeat(inv_freq, 2, axis=0)  # 2x for sin / cos

    radians = torch.einsum('bi,j -> bij', positions, inv_freq)  # [T, head_dim]
    radians = torch.view(radians, (batch_size, 1, seq_len, head_dim))
    return torch.cos(radians), torch.sin(radians)


def _splice_array(x: torch.Tensor) -> torch.Tensor:
    """Reorders the embedding dimension of an array, to make rotation easier."""
    # head_dim is assumed to be constructed/padded so it's even
    assert x.shape[-1] % 2 == 0

    even_dims = x[..., ::2]
    odd_dims = x[..., 1::2]
    return torch.stack((-odd_dims, even_dims), dim=-1).view(x.shape)


def _apply_rotary_encoding(
    x: torch.Tensor,
    max_wavelength: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Applies the rotary embedding matrix to an input array.

    Computes R*x, the multiplication between the rotation matrix R, and input x.

    Args:
      x: Array of shape [B, T, num_heads, head_dim]
      max_wavelength: Maximum wavelength that will appear in sin/cosine waveforms.
        This specifies the maximum sequence length for identifying unique
        positions.
      positions: A [B, T] tensor of positions.
    Returns:
      Array of rotary encoded input, of shape [B, T, num_heads, head_dim].
    """
    # {cos, sin}_matrix are [B, 1, T, head_dim]
    cos_matrix, sin_matrix = _make_rotation_matrices(
        x, max_wavelength, positions)
    # [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
    x = torch.moveaxis(x, -2, -3)
    # Apply the rotation.
    rotary_embeddings = x * cos_matrix + _splice_array(x) * sin_matrix
    # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim]
    return torch.moveaxis(rotary_embeddings, -3, -2)


def _apply_rotary_encoding_to_subset(
    x: torch.Tensor,
    fraction_to_rotate: float,
    fraction_heads_to_rotate: float,
    max_wavelength: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Applies a rotary positional encoding to a subset of dimensions."""
    if fraction_to_rotate > 1.0 or fraction_to_rotate <= 0.0:
        raise ValueError(
            f'fraction_to_rotate must be in (0, 1], got {fraction_to_rotate}.')
    _, _, num_heads, dim_per_head = x.shape

    def _to_even(x):
        return math.floor(x / 2.) * 2

    num_rotated_channels = _to_even(dim_per_head * fraction_to_rotate)

    num_rotated_heads = math.floor(fraction_heads_to_rotate * num_heads)

    if num_rotated_heads != num_heads:
        x_unrotated = x[..., num_rotated_heads:, :]
        x = x[..., :num_rotated_heads, :]

    if num_rotated_channels == dim_per_head:
        x = _apply_rotary_encoding(x, max_wavelength, positions)
    else:
        x_r = x[..., :num_rotated_channels]
        x_p = x[..., num_rotated_channels:]
        x_r = _apply_rotary_encoding(x_r, max_wavelength, positions)
        x = torch.concat((x_r, x_p), dim=-1)

    if num_rotated_heads != num_heads:
        x = torch.concat((x, x_unrotated), dim=-2)
    return x


#  -----------------------------------------------------------
#  -----------------------  Modules  -------------------------
#  -----------------------------------------------------------


class TrainablePositionEncoding(nn.Module):
    """Trainable position encoding."""

    def __init__(self, index_dim, num_channels, init_scale=0.02, name=None):
        super(TrainablePositionEncoding, self).__init__(name=name)
        self._index_dim = index_dim
        self._num_channels = num_channels
        self._init_scale = init_scale
        self.pos_embs = nn.Parameter(torch.empty(
            self._index_dim, self._num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        weight_init.trunc_normal_(self.pos_embs, stddev=self._init_scale)

    def __call__(self, batch_size, device):
        # If inputs shape > 2, broadcast to batch.
        if batch_size is not None:
            pos_embs = torch.broadcast_to(
                self.pos_embs, (batch_size,) + pos_embs.shape)
        return pos_embs.to(device)


@dataclass
class AttentionState:
    """State of the Attention module."""
    k: torch.Tensor  # [B, T, num_heads, head_dim]
    v: torch.Tensor  # [B, T, num_heads, head_dim]
    kv_positions: Optional[torch.Tensor]  # [B, T]
    memory_mask: Optional[torch.Tensor]  # [B, T]

def dynamic_slice(x, start_indices, slice_sizes):
    slice_list = [slice(i, s) for i, s in zip(start_indices, slice_sizes)]
    return x[slice_list]


class Attention(nn.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(self,
                 in_channels,
                 dropout_prob,
                 position_encoding_type,
                 fraction_to_rotate,
                 max_wavelength,
                 num_heads=8,
                 init_scale=1.0,
                 with_final_bias=True,
                 final_init_scale_multiplier=1.,
                 channels_per_head=None,
                 qkv_multi_head=False,
                 qk_channels=None,
                 v_channels=None,
                 output_channels=None,
                 fraction_heads_to_rotate=1.0):
        super(Attention, self).__init__()
        self._num_heads = num_heads
        self._init_scale = init_scale
        self._with_final_bias = with_final_bias
        self._final_init_scale = final_init_scale_multiplier * init_scale
        self._dropout_prob = dropout_prob
        self._qkv_multi_head = qkv_multi_head

        # If none of these are passed, the Q input determines the output shape:
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._output_channels = output_channels

        self._position_encoding_type = position_encoding_type
        self._fraction_to_rotate = fraction_to_rotate
        self._fraction_heads_to_rotate = fraction_heads_to_rotate
        self._max_wavelength = max_wavelength

        self.to_q = conv_1d(
            in_channels=self._in_channels,
            out_channels=self._qk_channels,
            init_scale=self._init_scale
        )
        self.to_k = conv_1d(
            in_channels=self._in_channels,
            out_channels=self._qk_channels,
            init_scale=self._init_scale
        )
        self.to_v = conv_1d(
            in_channels=self._v_channels,
            out_channels=self._v_channels,
            init_scale=self._init_scale
        )

    def _rotary_position_embeddings(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
        use_bias: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention using rotary encodings."""
        head_dim = q.shape[-1]
        rotary_queries = _apply_rotary_encoding_to_subset(
            q, self._fraction_to_rotate, self._fraction_heads_to_rotate,
            self._max_wavelength, q_positions)
        rotary_keys = _apply_rotary_encoding_to_subset(
            k, self._fraction_to_rotate, self._fraction_heads_to_rotate,
            self._max_wavelength, kv_positions)

        if use_bias:
            rotary_bias = self._multihead_bias(head_dim, 'rotary_bias')
            rotary_queries += rotary_bias
        return rotary_queries, rotary_keys

    def _attend(self, q, k, v, mask: Optional[torch.Tensor],
                q_positions: Optional[torch.Tensor],
                kv_positions: Optional[torch.Tensor],
                is_cross_attend: bool,
                is_training: bool):
        """Computes multi-head attention using a query, key and value.

        Args:
          q: Query with shape [batch, q_indices, num_heads, head_dim].
          k: Key with shape [batch, kv_indices, num_heads, head_dim].
          v: Value with shape [batch, kv_indices, num_heads, head_dim].
          mask: optional attention mask.
          q_positions: A [batch, q_indices] tensor of query positions for
            rotary attention.
          kv_positions: A [batch, kv_indices] tensor of key/value positions
            for rotary attention.
          is_cross_attend: whether the queries and keys come from the same array.
          is_training: whether this is used in a training context. If not, attention
            masks will be saved as state.
        Returns:
          Output of the attention with shape [batch, q_indices, hiddens]
        """
        batch, q_indices, num_heads, head_dim = q.shape
        hiddens = num_heads * head_dim

        if self._position_encoding_type == 'absolute':
            attention = torch.einsum('b t h d, b T h d -> b h t T', q, k)
        elif self._position_encoding_type == 'rotary':
            if q_positions is None or kv_positions is None:
                raise ValueError(
                    'Both q_positions and kv_positions must be specified '
                    'for rotary attention.')
            rotary_queries, rotary_keys = self._rotary_position_embeddings(
                q, k, q_positions, kv_positions)
            attention = torch.einsum(
                'b t h d, b T h d -> b h t T', rotary_queries, rotary_keys)
        else:
            raise ValueError('Invalid position encoding type.')

        scale = 1. / math.sqrt(head_dim)
        attention *= scale
        if mask is not None:
            mask = torch.broadcast_to(
                mask, (mask.shape[0],) + (num_heads,) + mask.shape[2:])
            assert mask.shape == attention.shape
            # Mask values of 0.0 indicate that an entry will be masked.
            attention = torch.where(mask, attention, -1e30)

        # Uncomment these for attention analysis.
        # They use extra memory, so leaving off for now.
        # if not is_training:
        #   hk.set_state('attention', attention)
        normalized = F.softmax(attention)
        # if not is_training:
        #   hk.set_state('attention_normalized', normalized)
        if is_training:
            normalized = F.dropout(self._dropout_prob, self.training)
        summed = torch.einsum('b h t T,b T h d->b t h d', normalized, v)
        return torch.view(summed, [batch, q_indices, hiddens])

    def _query_chunk_attention(self,
                               query,
                               key,
                               value,
                               mask,
                               precision,
                               is_training: bool,
                               key_chunk_size: int = 4096):
        """Multi-head dot product attention with a limited number of queries."""
        num_kv, k_features = key.shape
        v_features = value.shape[-1]

        qk_channels_per_head = k_features // self._num_heads
        v_channels_per_head = v_features // self._num_heads

        key_chunk_size = min(key_chunk_size, num_kv)
        query = query / torch.sqrt(qk_channels_per_head)

        @functools.partial(torch.utils.checkpoint)
        def summarize_chunk(query, key, value, mask, dropout_rng):
            query = query.view(
                query.shape[0], self._num_heads, qk_channels_per_head)
            key = key.view(
                key.shape[0], self._num_heads, qk_channels_per_head)
            value = value.view(
                value.shape[0], self._num_heads, v_channels_per_head)

            attn_weights = torch.einsum(
                'q hd, k h d -> q h k', query, key, precision=precision)
            mask = torch.broadcast_to(torch.moveaxis(
                mask, 0, 1), attn_weights.shape)
            attn_weights = torch.where(mask, attn_weights, -1e30)
            max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_weights = F.dropout(exp_weights, self._dropout_prob, self.training)
            exp_values = torch.einsum(
                'v h f, q h v -> q h f', value, exp_weights, precision=precision)
            return (exp_values, exp_weights.sum(axis=-1),
                    max_score.view((query.shape[0], self._num_heads)))

        def chunk_scanner(chunk_idx):
            key_chunk = dynamic_slice(
                key, (chunk_idx, 0),
                slice_sizes=(key_chunk_size, k_features))
            value_chunk = dynamic_slice(
                value, (chunk_idx, 0),
                slice_sizes=(key_chunk_size, v_features))
            mask_chunk = dynamic_slice(
                mask, (0, 0, chunk_idx),
                slice_sizes=(1, query.shape[0], key_chunk_size))
            return summarize_chunk(
                query, key_chunk, value_chunk, mask_chunk, hk.next_rng_key())

        _, (chunk_values, chunk_weights, chunk_max) = scan(
            lambda _, x: ((), chunk_scanner(x)), (),
            xs=torch.arange(0, num_kv, key_chunk_size))

        global_max = torch.max(chunk_max, dim=0, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= max_diffs.unsqueeze(-1)
        chunk_weights *= max_diffs

        all_values = chunk_values.sum(dim=0)
        all_weights = chunk_weights.unsqueeze(-1).sum(dim=0)
        all_values /= all_weights
        return all_values.view(
            all_values.shape[0], all_values.shape[1] * all_values.shape[2])

    def _chunked_attend(self,
                        q,
                        k,
                        v,
                        mask: Optional[torch.Tensor],
                        q_positions: Optional[torch.Tensor],
                        kv_positions: Optional[torch.Tensor],
                        is_training: bool,
                        precision: torch.float32 = torch.float32,
                        query_chunk_size: int = 1024,
                        key_chunk_size: int = 4096):
        """Memory-efficient multi-head dot product attention."""
        # Implementation adapted from Self-Attention Does Not Need O(n^2) Memory
        # Rabe and Staats, https://arxiv.org/pdf/2112.05682.pdf
        if self._position_encoding_type == 'absolute':
            pass
        elif self._position_encoding_type == 'rotary':
            if q_positions is None or kv_positions is None:
                raise ValueError(
                    'Both q_positions and kv_positions must be specified '
                    'for rotary attention.')
            q, k = self._rotary_position_embeddings(
                q, k, q_positions, kv_positions)
        else:
            raise ValueError('Invalid position encoding type.')

        if mask is None:
            mask = torch.ones([q.shape[0], 1, q.shape[1], k.shape[1]])

        # Split heads were required for rotary position encoding above.
        # Now combine the heads so that the shapes are less likely to require TPU
        # padding as we iterate through query and key chunks.
        q = torch.reshape(q, [q.shape[0], q.shape[1], q.shape[2] * q.shape[3]])
        k = torch.reshape(k, [k.shape[0], k.shape[1], k.shape[2] * k.shape[3]])
        v = torch.reshape(v, [v.shape[0], v.shape[1], v.shape[2] * v.shape[3]])

        def attn(q, k, v, mask):
            num_q, q_features = q.shape

            def chunk_scanner(chunk_idx, _):
                query_chunk = dynamic_slice(
                    q, (chunk_idx, 0),
                    slice_sizes=(min(query_chunk_size, num_q), q_features))
                mask_chunk = dynamic_slice(
                    mask, (0, chunk_idx, 0),
                    slice_sizes=(mask.shape[0],
                                 min(query_chunk_size, num_q),
                                 k.shape[0]))
                return (chunk_idx + query_chunk_size,
                        self._query_chunk_attention(
                            query_chunk, k, v, mask_chunk,
                            key_chunk_size=key_chunk_size, precision=precision,
                            is_training=is_training))

            _, res = scan(
                chunk_scanner,
                init=0,
                xs=None,
                length=math.ceil(num_q / query_chunk_size))
            return res.reshape(num_q, v.shape[-1])
        return vmap(attn)(q, k, v, mask)

    def forward(self,
                inputs_q,
                inputs_kv,
                is_cross_attend: bool,
                is_training: bool,
                memory_type: str,
                mask: Optional[jnp.ndarray] = None,
                memory: Optional[AttentionState] = None,
                q_positions: Optional[jnp.ndarray] = None,
                kv_positions: Optional[jnp.ndarray] = None,
                head_group_size: int = 0,
                use_chunked_attention: bool = False,
                query_chunk_size: int = 1024,
                key_chunk_size: int = 4096):
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if self._qk_channels is None:
            self._qk_channels = inputs_q.shape[-1]
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if self._v_channels is None:
            self._v_channels = self._qk_channels
        # Project the output of QKV attention to a desired number of channels.
        # Default to the same number as the output of the QKV attention operation.
        if self._output_channels is None:
            self._output_channels = self._v_channels

        assert self._qk_channels % self._num_heads == 0
        assert self._v_channels % self._num_heads == 0
        qk_channels_per_head = self._qk_channels // self._num_heads
        v_channels_per_head = self._v_channels // self._num_heads

        # -----------------------------
        # ------ Compute Q, K, V ------
        # -----------------------------
        # Project QKV to a common feature dimension.
        q = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_q)
        k = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_kv)
        v = conv_1d(self._v_channels, init_scale=self._init_scale)(inputs_kv)

        # Reshape channels for multi-head attention.
        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape

        q = jnp.reshape(
            q, [batch, q_time, self._num_heads, qk_channels_per_head])
        k = jnp.reshape(
            k, [batch, kv_time, self._num_heads, qk_channels_per_head])
        v = jnp.reshape(
            v, [batch, kv_time, self._num_heads, v_channels_per_head])

        # --------------------
        # ------ Memory ------
        # --------------------
        def _update_memory(mem_arr, new_vals):
            # Add new memories to the right, possibly overwriting the oldest memories.
            # e.g.
            # [0, 0, 0] <- [0, 0 mem_0] or
            # [mem_0, mem_1, mem_2] <- [mem_1, mem_2, mem_3] or
            # [mem_0, mem_1, mem_2] <- [mem_2, mem_3, mem_4]
            num_new_vals = new_vals.shape[1]
            assert num_new_vals <= mem_arr.shape[1]
            mem_arr = jnp.roll(mem_arr, axis=1, shift=-num_new_vals)
            return jax.lax.dynamic_update_slice_in_dim(
                mem_arr, new_vals, start_index=-num_new_vals, axis=1)

        # Grab additional attention targets from memory.
        if memory is None:
            memory_mask = None
        else:
            # We assume that masking is not required when caching, i.e. that the
            # current prediction is subsequent to all inputs.
            if memory_type == 'none':
                raise ValueError(
                    'Memory input not expected when memory_type is `none`.')
            elif memory_type == 'kv':
                assert mask is None
                k = jnp.concatenate([memory.k, k], axis=1)
                v = jnp.concatenate([memory.v, v], axis=1)
                if kv_positions is not None:
                    kv_positions = jnp.concatenate(
                        [memory.kv_positions, kv_positions], axis=1)
                memory_mask = memory.memory_mask
                if memory_mask is not None:
                    memory_mask = jnp.concatenate(
                        [memory_mask,
                         jnp.ones([batch, memory.k.shape[1]])], axis=-1)
            elif memory_type == 'fixed_size_kv':
                assert memory.memory_mask is not None
                k = _update_memory(memory.k, k)
                v = _update_memory(memory.v, v)
                if kv_positions is not None:
                    kv_positions = _update_memory(
                        memory.kv_positions, kv_positions)
                memory_mask = _update_memory(
                    memory.memory_mask,
                    jnp.ones([batch, kv_time], dtype=jnp.float32))
                attn_mask = flax.linen.make_attention_mask(
                    query_input=jnp.ones([batch, q_time], dtype=jnp.float32),
                    key_input=memory_mask)
                if mask is None:
                    mask = attn_mask
                else:
                    mask = flax.linen.combine_masks(attn_mask, mask)
            else:
                raise ValueError(f'Unknown memory_type: {memory_type}')

        # ------------------------------
        # ------ Attention -> MLP ------
        # ------------------------------
        if head_group_size:
            assert not use_chunked_attention
            # Attention maps are [n_heads, q_indices, kv_indices], so memory usage
            # grows linearly with the number of heads regardless of embedding sizes.
            # However, this is only a temporary allocation for the softmax.
            # This option computes heads in smaller groups, trading compute for
            # memory.
            # Note that finding the right option here usually requires some trial and
            # error because the combination of XLA optimizations and TPU padding means
            # it's not always clear what configuration will actually use less memory
            # in practice. In general, the largest group that works is best.
            per_head_results = []
            assert self._num_heads % head_group_size == 0
            for i in range(0, self._num_heads, head_group_size):
                per_head_result = self._attend(
                    q[:, :, i:i + head_group_size],
                    k[:, :, i:i + head_group_size],
                    v[:, :, i:i + head_group_size],
                    mask=mask,
                    q_positions=q_positions,
                    kv_positions=kv_positions,
                    is_cross_attend=is_cross_attend,
                    is_training=is_training)
                per_head_results.append(per_head_result)
            result = jnp.concatenate(per_head_results, axis=-1)
        else:
            if use_chunked_attention:
                result = self._chunked_attend(q, k, v,
                                              mask=mask,
                                              q_positions=q_positions,
                                              kv_positions=kv_positions,
                                              is_training=is_training,
                                              query_chunk_size=query_chunk_size,
                                              key_chunk_size=key_chunk_size)
            else:
                result = self._attend(q, k, v,
                                      mask=mask,
                                      q_positions=q_positions,
                                      kv_positions=kv_positions,
                                      is_cross_attend=is_cross_attend,
                                      is_training=is_training)

        outputs = conv_1d(
            self._output_channels,
            with_bias=self._with_final_bias,
            init_scale=self._final_init_scale)(result)

        if memory_type == 'none':
            memory = None
        else:
            memory = AttentionState(
                k=k,
                v=v,
                kv_positions=kv_positions,
                memory_mask=memory_mask)

        return outputs, memory
