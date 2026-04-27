import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _remap_pp_weight_name(weight_name: str, layer_start: int, layer_end: int):
    """Return the remapped param name for this PP rank, or None to skip."""
    m = _LAYER_RE.search(weight_name)
    if m:
        layer_idx = int(m.group(1))
        if layer_idx < layer_start or layer_idx >= layer_end:
            return None
        return weight_name.replace(f"layers.{layer_idx}.", f"layers.{layer_idx - layer_start}.")
    return weight_name


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    inner = getattr(model, "model", model)
    layer_start = getattr(inner, "layer_start", 0)
    layer_end = getattr(inner, "layer_end", float("inf"))
    is_first_rank = getattr(inner, "is_first_rank", True)
    is_last_rank = getattr(inner, "is_last_rank", True)

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if "embed_tokens" in weight_name:
                    if is_first_rank:
                        pass
                    elif is_last_rank and hasattr(model, "lm_head"):
                        mapped_name = weight_name.replace("model.embed_tokens", "lm_head")
                        param = model.get_parameter(mapped_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                        continue
                    else:
                        continue
                if ("norm." in weight_name or "lm_head" in weight_name) and not is_last_rank:
                    continue

                mapped_name = _remap_pp_weight_name(weight_name, layer_start, layer_end)
                if mapped_name is None:
                    continue

                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = mapped_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        module_path = param_name.rsplit(".", 1)[0]
                        module = model.get_submodule(module_path)
                        module.weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(mapped_name)
                    default_weight_loader(param, f.get_tensor(weight_name))
