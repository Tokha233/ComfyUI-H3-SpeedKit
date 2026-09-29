#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Materialize Comfy's actual patched INT8 ConvRot weights once, then save.

Uses public model-patcher methods. Does not add low-rank matrices to INT8 bytes,
change alpha rules, or install a global patch. Requires CPU RAM for the model
and the materialized checkpoint, plus enough VRAM for the largest layer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfyui', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--lora', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--strength', type=float, default=1.0)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new filename')
    sys.path.insert(0, str(a.comfyui.resolve()))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = ['speedkit-merge', '--disable-cuda-malloc']
    import main as comfy_main
    import torch
    import comfy.sd
    import comfy.utils
    from comfy_kitchen.tensor.base import QuantizedTensor
    from safetensors import safe_open
    torch.set_num_threads(4)
    with torch.inference_mode():
        model = comfy.sd.load_diffusion_model(str(a.base), model_options={})
        model, _ = comfy.sd.load_lora_for_models(model, None,
            comfy.utils.load_torch_file(str(a.lora), safe_load=True), a.strength, 0.0)
        if not model.patches:
            raise ValueError('No LoRA keys matched the selected model')
        module = model.model.diffusion_model
        state = module.state_dict()
        handled = set()
        output = {}
        for key, value in state.items():
            if key in handled:
                continue
            if key.endswith(('.weight', '.bias')):
                weight = comfy.utils.get_attr(module, key)
                if isinstance(weight, QuantizedTensor):
                    patched = model.patch_weight_to_device('diffusion_model.' + key,
                        device_to=model.load_device, return_weight=True)
                    material = patched.state_dict(key)
                    output.update({k: t.to('cpu').contiguous() for k, t in material.items()})
                    handled.update(material)
                    del patched, material
                    continue
                if 'diffusion_model.' + key in model.patches:
                    value = model.patch_weight_to_device('diffusion_model.' + key,
                        device_to=model.load_device, return_weight=True)
            output[key] = value.detach().to('cpu').contiguous()
        with safe_open(str(a.base), framework='pt') as f:
            metadata = dict(f.metadata() or {})
        metadata['h3_speedkit_merge'] = json.dumps({'strength': a.strength, 'adapter': a.lora.name,
            'base': a.base.name, 'method': 'Comfy ModelPatcher.patch_weight_to_device'})
        a.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.output.with_suffix('.partial.safetensors')
        try:
            comfy.utils.save_torch_file(output, str(temporary), metadata=metadata)
            temporary.replace(a.output)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {'tensors': len(output), 'patches': len(model.patches), 'strength': a.strength,
                    'sha256': hashlib.file_digest(a.output.open('rb'), 'sha256').hexdigest(),
                    'base_sha256': hashlib.file_digest(a.base.open('rb'), 'sha256').hexdigest(),
                    'lora_sha256': hashlib.file_digest(a.lora.open('rb'), 'sha256').hexdigest()}
        a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
