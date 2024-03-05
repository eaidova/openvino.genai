from dataclasses import dataclass
import torch
import functools

def slice0d(x, start, end):
    return x[start:end, ...]

def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start:int, end:int):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


DIM_TO_SLICE = {
    0: slice0d,
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

@torch.jit.script_if_tracing
def get_slice_end(key:torch.Tensor, sink_window_size, k_seq_dim, shift, cache_size, sink_size):
    sink_size  = torch.maximum(key.shape[k_seq_dim] - sink_window_size + shift, sink_size)
    return torch.minimum(sink_size, torch.tensor(key.shape[k_seq_dim]))

@dataclass
class AttentionSinkKVCache:
    attention_sink_size: int = 4
    attention_sink_window_size: int = 1020
    k_seq_dim: int = 2
    v_seq_dim: int = 2
    attention_sink_shift: int = 7

    def __post_init__(self):
        self.cache_size = self.attention_sink_size + self.attention_sink_window_size
        self.k_slice = DIM_TO_SLICE[self.k_seq_dim]
        self.v_slice = DIM_TO_SLICE[self.v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        slice_end = get_slice_end(past_key_values[0][0], torch.tensor(self.attention_sink_window_size), torch.tensor(self.k_seq_dim), torch.tensor(self.attention_sink_shift), torch.tensor(self.cache_size), torch.tensor(self.attention_sink_size))
        seq_len = past_key_values[0][0].shape[self.k_seq_dim]
        return tuple(
            tuple(
                [torch.cat([
                    self.k_slice(k, 0, min(seq_len, self.attention_sink_size)), 
                    self.k_slice(k, slice_end, seq_len)
                ], dim=self.k_seq_dim),

                torch.cat([
                    self.v_slice(v, 0, self.attention_sink_size), 
                    self.v_slice(v, slice_end, seq_len)
                ], dim=self.v_seq_dim),
            ])
            for k, v in past_key_values
        )

def attention_sink_patch(model, sink_size=4, window_size=1020, shift=7):
    forward = model.forward
    k_dim = 2
    v_dim = 2
    model_type = model.config.model_type
    if model_type == "bloom":
        v_dim = 1
    elif model_type == "qwen":
        v_dim = 1
        k_dim = 1
    elif model_type == "chatglm":
        v_dim = 0
        k_dim = 0
    sink = AttentionSinkKVCache(sink_size, window_size, k_dim, v_dim, shift)
    model.config.attention_sink = (sink_size, window_size, shift)
    
    
    @functools.wraps(forward)
    def as_patched_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        pkv = result.past_key_values
        pkv  = sink(pkv)
        result.past_key_values = pkv
        return result
    model._orig_forward = forward
    model.forward = as_patched_forward
    return model