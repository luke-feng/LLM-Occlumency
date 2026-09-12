#!/usr/bin/env python3
"""Optional single-device local SFT reference, not original weight reproduction.

Caller supplies a local safetensors checkpoint and ordinary prompt/response data.
--validate-only and --help never import frameworks, load weights or write output.
The reference does not generate training data or run an evaluation/inference loop.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from models.loading import load_local_base, load_local_tokenizer, thinking_kwargs, validate_loading, validate_device_contract

TARGET_MODULES=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')


@dataclass
class SFTConfig:
    model_dir: str
    train_jsonl: str
    outdir: str
    extra_jsonl: str = ''
    seed: int = 0
    max_length: int = 512
    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = TARGET_MODULES
    gradient_checkpointing: bool = True
    precision: str = 'auto'
    device: str = 'cpu'
    load_in_4bit: bool = False


def output_path(value, model_path):
    if not isinstance(value,(str,Path)) or not str(value).strip():
        raise ValueError('outdir is required')
    out=Path(value).expanduser().resolve()
    if out==model_path or model_path in out.parents:
        raise ValueError('output must be outside the source model directory')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('output directory must be absent or empty; existing adapters are never overwritten')
    return out


def validate_config(config):
    model=validate_loading(config.model_dir,config.precision,config.device,config.load_in_4bit)
    for field in ('max_length','batch_size','gradient_accumulation_steps','lora_r','lora_alpha'):
        value=getattr(config,field)
        if type(value) is not int or value<=0:
            raise ValueError(field+' must be a positive integer')
    if type(config.seed) is not int or config.seed<0:
        raise ValueError('seed must be a nonnegative integer')
    for field in ('epochs','learning_rate'):
        value=getattr(config,field)
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError(field+' must be finite and positive')
    if type(config.lora_dropout) not in (int,float) or not math.isfinite(config.lora_dropout) or not 0<=config.lora_dropout<1:
        raise ValueError('lora_dropout must be in [0,1)')
    if type(config.gradient_checkpointing) is not bool:
        raise ValueError('gradient_checkpointing must be boolean')
    if not isinstance(config.target_modules,(list,tuple)) or not config.target_modules or not all(isinstance(x,str) and x.strip() for x in config.target_modules):
        raise ValueError('target_modules must be a nonempty sequence of names')
    if len(set(config.target_modules))!=len(config.target_modules):
        raise ValueError('duplicate target module')
    if config.precision=='float16' and not config.device.startswith('cuda'):
        raise ValueError('reference fp16 training requires an explicitly selected CUDA device')
    out=output_path(config.outdir,model)
    files=[Path(config.train_jsonl).expanduser().resolve()]
    if config.extra_jsonl:
        files.append(Path(config.extra_jsonl).expanduser().resolve())
    if len(set(files))!=len(files):
        raise ValueError('primary and supplemental input must be distinct files')
    for path in files:
        if not path.is_file():
            raise ValueError('training data must be an existing local JSONL file')
    return model,out,files


def read_pairs(paths):
    rows=[]
    input_digests=[]
    def unique(pairs):
        d={}
        for key,value in pairs:
            if key in d:
                raise ValueError('duplicate row field')
            d[key]=value
        return d
    for path in paths:
        raw=Path(path).read_bytes()
        input_digests.append(hashlib.sha256(raw).hexdigest())
        for number,line in enumerate(raw.decode('utf-8').splitlines(),1):
                if not line.strip():
                    continue
                try:
                    row=json.loads(line,object_pairs_hook=unique)
                    if not isinstance(row,dict) or set(row)!={'prompt','response'} or not all(isinstance(row[k],str) and row[k].strip() for k in row):
                        raise ValueError('expected nonempty prompt and response strings only')
                except (ValueError,TypeError) as exc:
                    # Never echo input contents, which may be private user data.
                    raise ValueError('invalid training row at line '+str(number)) from exc
                rows.append(row)
    if not rows:
        raise ValueError('training dataset is empty')
    return rows,input_digests


def chat_text(tokenizer,prompt,response=None):
    if getattr(tokenizer,'chat_template',None):
        messages=[{'role':'user','content':prompt}]
        if response is not None:
            messages.append({'role':'assistant','content':response})
        return tokenizer.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=response is None,**thinking_kwargs(tokenizer))
    if response is None:
        return 'User: '+prompt+'\nAssistant:'
    return 'User: '+prompt+'\nAssistant: '+response+(tokenizer.eos_token or '')


def encode_example(row,tokenizer,max_length):
    full=tokenizer(chat_text(tokenizer,row['prompt'],row['response']),truncation=True,max_length=max_length)
    prefix=tokenizer(chat_text(tokenizer,row['prompt']),truncation=True,max_length=max_length)
    ids=list(full['input_ids'])
    attention=list(full['attention_mask'])
    if not ids or len(ids)!=len(attention):
        raise ValueError('invalid tokenized training example')
    prompt_len=min(len(prefix['input_ids']),len(ids))
    labels=[-100]*prompt_len+ids[prompt_len:]
    if all(x==-100 for x in labels):
        raise ValueError('no supervised tokens remain after prefix masking/truncation')
    return {'input_ids':ids,'attention_mask':attention,'labels':labels}


def pad_examples(features,pad_token_id):
    if not features or type(pad_token_id) is not int:
        raise ValueError('nonempty features and integer pad token required')
    size=max(len(f['input_ids']) for f in features)
    result={'input_ids':[],'attention_mask':[],'labels':[]}
    for f in features:
        n=len(f['input_ids'])
        if not n or len(f['attention_mask'])!=n or len(f['labels'])!=n:
            raise ValueError('unaligned training feature')
        result['input_ids'].append(list(f['input_ids'])+[pad_token_id]*(size-n))
        result['attention_mask'].append(list(f['attention_mask'])+[0]*(size-n))
        result['labels'].append(list(f['labels'])+[-100]*(size-n))
    return result


class TensorCollator:
    def __init__(self,pad_token_id):self.pad_token_id=pad_token_id
    def __call__(self,features):
        padded=pad_examples(features,self.pad_token_id)
        import torch
        return {key:torch.tensor(value,dtype=torch.long) for key,value in padded.items()}


def validate_inputs(config):
    model,out,paths=validate_config(config)
    rows,input_digests=read_pairs(paths)
    return model,out,paths,rows,input_digests


def training_argument_values(config,out):
    """Pure constructor arguments: Trainer must honor the selected CPU mode."""
    return {'output_dir':str(out/'trainer_state'),'seed':config.seed,
        'num_train_epochs':config.epochs,'per_device_train_batch_size':config.batch_size,
        'gradient_accumulation_steps':config.gradient_accumulation_steps,
        'learning_rate':config.learning_rate,'logging_steps':5,'save_strategy':'no','report_to':[],
        'use_cpu':config.device=='cpu','fp16':config.precision=='float16',
        'bf16':config.precision=='bfloat16','remove_unused_columns':False}


def train(config):
    model_path,out,paths,rows,input_digests=validate_inputs(config)
    # Validate every supervised span before loading base weights or creating output.
    tokenizer=load_local_tokenizer(model_path)
    dataset=[encode_example(row,tokenizer,config.max_length) for row in rows]
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments
    world_size=int(os.environ.get('WORLD_SIZE','1'))
    validate_device_contract(config.device,torch.cuda.device_count() if config.device!='cpu' else 0,world_size)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model=load_local_base(model_path,config.precision,config.device,config.load_in_4bit)
    if config.load_in_4bit:
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=config.gradient_checkpointing)
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model,'enable_input_require_grads'):
            model.enable_input_require_grads()
        model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=config.lora_r,
        lora_alpha=config.lora_alpha,lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules)))
    # Recheck after potentially lengthy local loads. No reuse/overwrite of adapters.
    output_path(out,model_path)
    out.mkdir(parents=True,exist_ok=True)
    arguments=TrainingArguments(**training_argument_values(config,out))
    trainer=Trainer(model=model,args=arguments,train_dataset=dataset,
                    data_collator=TensorCollator(tokenizer.pad_token_id))
    result=trainer.train()
    model.save_pretrained(str(out),safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    parameters=asdict(config)
    for key in ('model_dir','train_jsonl','extra_jsonl','outdir'):
        parameters.pop(key)
    record={'schema_version':'generic_local_sft_run_v1','scope':'new caller-supplied reference run; not original paper weights',
            'parameters':parameters,'n_examples':len(rows),'training_loss':result.training_loss,
            'optimizer_steps':result.global_step,
            'input_sha256':input_digests}
    (out/'reference_training.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--train-jsonl',required=True)
    parser.add_argument('--extra-jsonl',default='')
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--validate-only',action='store_true',help='Validate local paths, configuration and JSONL without frameworks, tokenization or output writes')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-length',type=int,default=512)
    parser.add_argument('--epochs',type=float,default=3.0)
    parser.add_argument('--learning-rate',type=float,default=2e-4)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--gradient-accumulation-steps',type=int,default=8)
    parser.add_argument('--lora-r',type=int,default=16)
    parser.add_argument('--lora-alpha',type=int,default=32)
    parser.add_argument('--lora-dropout',type=float,default=.05)
    parser.add_argument('--target-modules',default=','.join(TARGET_MODULES))
    parser.add_argument('--precision',choices=['auto','float32','float16','bfloat16'],default='auto')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--load-in-4bit',action='store_true')
    parser.add_argument('--no-gradient-checkpointing',action='store_true')
    args=parser.parse_args()
    values=vars(args)
    validate_only=values.pop('validate_only')
    values['target_modules']=tuple(x.strip() for x in values['target_modules'].split(',') if x.strip())
    values['gradient_checkpointing']=not values.pop('no_gradient_checkpointing')
    config=SFTConfig(**values)
    if validate_only:
        _,_,_,rows,_=validate_inputs(config)
        print(json.dumps({'valid':True,'n_examples':len(rows),'frameworks_loaded':False,
                          'tokenization_validated':False,'output_created':False}))
    else:
        print(json.dumps(train(config),indent=2))


if __name__=='__main__':
    main()
