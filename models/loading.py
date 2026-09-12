"""Generic local checkpoint loading, separated from every experiment workload.

No inference CLI, remote code, hub identifier, download fallback or search helper.
Frameworks are imported only inside explicit loading functions, not by validation.
"""
import json
from pathlib import Path


def local_directory(value, label):
    if not isinstance(value,(str,Path)) or not str(value).strip():
        raise ValueError(label+' must name an existing local directory')
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(label+' must name an existing local directory, not a hub ID or URL')
    return path


def local_model_directory(value):
    path = local_directory(value,'model')
    if not (path/'config.json').is_file():
        raise ValueError('local model directory lacks config.json')
    if not any((path/name).is_file() for name in ('model.safetensors','model.safetensors.index.json')):
        raise ValueError('reference loader requires local safetensors weights or index')
    return path


def local_adapter_directory(value):
    path = local_directory(value,'adapter')
    if not (path/'adapter_config.json').is_file() or not (path/'adapter_model.safetensors').is_file():
        raise ValueError('reference adapter requires local configuration and safetensors weights')
    config = json.loads((path/'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('peft_type')!='LORA' or config.get('auto_mapping'):
        raise ValueError('only standard local LoRA adapters without custom auto-mapping are supported')
    return path


def validate_loading(model_dir, precision='auto', device='cpu', load_in_4bit=False):
    path = local_model_directory(model_dir)
    if precision not in {'auto','float32','float16','bfloat16'}:
        raise ValueError('unsupported precision')
    if device not in {'cpu','cuda','cuda:0'}:
        raise ValueError('reference supports CPU or one visible logical CUDA device zero only')
    if type(load_in_4bit) is not bool:
        raise ValueError('load_in_4bit must be boolean')
    if load_in_4bit and not device.startswith('cuda'):
        raise ValueError('the optional NF4 reference path requires an explicitly selected CUDA device')
    return path


def validate_device_contract(device,visible_cuda_devices=0,world_size=1):
    if device not in {'cpu','cuda','cuda:0'} or type(world_size) is not int or world_size!=1:
        raise ValueError('reference is single-process CPU or single-visible-CUDA0 only')
    if device!='cpu' and (type(visible_cuda_devices) is not int or visible_cuda_devices!=1):
        raise ValueError('CUDA reference requires exactly one visible device, addressed as logical CUDA0')


def thinking_kwargs(tokenizer):
    template = getattr(tokenizer,'chat_template',None)
    return {'enable_thinking':False} if isinstance(template,str) and 'enable_thinking' in template else {}


def load_local_tokenizer(model_dir):
    path = local_model_directory(model_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path),use_fast=True,
                                             local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('tokenizer needs a pad token or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_local_base(model_dir, precision='auto', device='cpu', load_in_4bit=False):
    path = validate_loading(model_dir,precision,device,load_in_4bit)
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    validate_device_contract(device,torch.cuda.device_count() if device!='cpu' else 0)
    dtype = 'auto' if precision=='auto' else getattr(torch,precision)
    kwargs = {'local_files_only':True,'trust_remote_code':False,
              'use_safetensors':True,'torch_dtype':dtype}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs['device_map'] = {'':device}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path),**kwargs)
    except ValueError as exc:
        # Only the recognized non-causal configuration case gets another head.
        # Both attempts remain local-only; architecture/version failures propagate.
        if 'Unrecognized configuration class' not in str(exc) or 'AutoModelForCausalLM' not in str(exc):
            raise
        model = AutoModelForImageTextToText.from_pretrained(str(path),**kwargs)
    if not load_in_4bit:
        model = model.to(device)
    return model


def load_local_adapter(model, adapter_dir):
    path = local_adapter_directory(adapter_dir)
    from peft import PeftModel
    return PeftModel.from_pretrained(model,str(path),local_files_only=True,is_trainable=False)
