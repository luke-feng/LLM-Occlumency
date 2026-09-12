"""Pure tests with synthetic strings/token IDs; never import model frameworks."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from analysis.check_package import check_imports
from models.loading import local_adapter_directory, thinking_kwargs, validate_device_contract
from models.lora_sft import (SFTConfig,encode_example,pad_examples,read_pairs,
                             training_argument_values,validate_inputs)


class ToyTokenizer:
    chat_template=None
    eos_token='!'
    def __call__(self,text,truncation,max_length):
        ids=[ord(c) for c in text][:max_length]
        return {'input_ids':ids,'attention_mask':[1]*len(ids)}


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.model=self.root/'local-model'
        self.model.mkdir()
        (self.model/'config.json').write_text('{}')
        # A zero-byte sentinel is never loaded; this test validates paths only.
        (self.model/'model.safetensors').write_bytes(b'')
        self.data=self.root/'train.jsonl'
        self.data.write_text('{"prompt":"hello","response":"thanks"}\n')
        self.out=self.root/'new-adapter'
        self.config=SFTConfig(str(self.model),str(self.data),str(self.out))
    def tearDown(self):self.temporary.cleanup()

    def test_validation_does_not_create_output_or_load_frameworks(self):
        before=set(sys.modules)
        *_,rows,digests=validate_inputs(self.config)
        self.assertEqual(len(rows),1)
        self.assertEqual(len(digests),1)
        self.assertFalse(self.out.exists())
        self.assertFalse({'torch','transformers','peft','accelerate','bitsandbytes'}&(set(sys.modules)-before))

    def test_input_hashes_bind_parsed_bytes_not_later_file(self):
        original=self.data.read_bytes()
        *_,rows,digests=validate_inputs(self.config)
        self.data.write_text('{"prompt":"changed","response":"later"}\n')
        self.assertEqual(digests,[hashlib.sha256(original).hexdigest()])
        self.assertEqual(rows[0]['prompt'],'hello')
        self.assertNotEqual(digests[0],hashlib.sha256(self.data.read_bytes()).hexdigest())

    def test_output_nonempty_or_within_model_refused(self):
        self.out.mkdir()
        old=self.out/'adapter_config.json'
        old.write_text('keep')
        with self.assertRaisesRegex(ValueError,'absent or empty'):
            validate_inputs(self.config)
        self.assertEqual(old.read_text(),'keep')
        with self.assertRaisesRegex(ValueError,'outside the source model'):
            validate_inputs(replace(self.config,outdir=str(self.model/'new')))

    def test_invalid_data_never_creates_output(self):
        for text in ['{}','{"prompt":"a","response":""}','{"prompt":"a","response":"b","extra":1}',
                     '{"prompt":"a","prompt":"b","response":"c"}','not JSON','']:
            self.data.write_text(text)
            with self.subTest(text=text),self.assertRaises(ValueError):
                validate_inputs(self.config)
            self.assertFalse(self.out.exists())

    def test_hub_id_or_url_is_not_a_local_checkpoint(self):
        for name in ['a-nonexistent-hub-model','https://example.invalid/model']:
            with self.subTest(name=name),self.assertRaises(ValueError):
                validate_inputs(replace(self.config,model_dir=name))

    def test_missing_weights_and_duplicate_input_refused(self):
        with self.assertRaisesRegex(ValueError,'distinct'):
            validate_inputs(replace(self.config,extra_jsonl=str(self.data)))
        (self.model/'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError,'safetensors'):
            validate_inputs(self.config)

    def test_invalid_numeric_configuration(self):
        for field,value in [('epochs',float('nan')),('lora_r',True),('lora_dropout',1.0),
                            ('gradient_accumulation_steps',0),('target_modules',('q_proj','q_proj'))]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_inputs(replace(self.config,**{field:value}))

    def test_token_masking_and_padding(self):
        toy=ToyTokenizer()
        encoded=encode_example({'prompt':'hi','response':'thanks'},toy,512)
        prefix=len('User: hi\nAssistant:')
        self.assertEqual(encoded['labels'][:prefix],[-100]*prefix)
        self.assertEqual(encoded['labels'][prefix:],encoded['input_ids'][prefix:])
        small=encode_example({'prompt':'x','response':'y'},toy,512)
        padded=pad_examples([encoded,small],0)
        self.assertEqual(len(padded['labels'][0]),len(padded['labels'][1]))
        self.assertEqual(padded['labels'][1][-1],-100)
        self.assertEqual(padded['attention_mask'][1][-1],0)

    def test_no_supervised_token_is_rejected_not_unmasked(self):
        with self.assertRaisesRegex(ValueError,'no supervised tokens'):
            encode_example({'prompt':'long enough prompt','response':'ok'},ToyTokenizer(),3)

    def test_optional_thinking_flag_is_template_gated(self):
        toy=ToyTokenizer()
        self.assertEqual(thinking_kwargs(toy),{})
        toy.chat_template='enable_thinking'
        self.assertEqual(thinking_kwargs(toy),{'enable_thinking':False})

    def test_device_argument_contract(self):
        cpu=training_argument_values(self.config,self.out)
        self.assertIs(cpu['use_cpu'],True)
        self.assertIs(cpu['fp16'],False)
        gpu=training_argument_values(replace(self.config,device='cuda',precision='bfloat16'),self.out)
        self.assertIs(gpu['use_cpu'],False)
        self.assertIs(gpu['bf16'],True)
        validate_device_contract('cpu',0,1)
        validate_device_contract('cuda:0',1,1)
        for device,count,world in [('mps',0,1),('cuda:1',1,1),('cuda',2,1),('cuda',0,1),('cpu',0,2)]:
            with self.subTest(device=device,count=count,world=world),self.assertRaises(ValueError):
                validate_device_contract(device,count,world)

    def test_unimplemented_devices_and_cpu_nf4_refused(self):
        for changes in [{'device':'mps'},{'device':'cuda:2'},{'load_in_4bit':True},{'precision':'float16'}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_inputs(replace(self.config,**changes))

    def test_adapter_validation_without_loading(self):
        adapter=self.root/'adapter'
        adapter.mkdir()
        (adapter/'adapter_model.safetensors').write_bytes(b'')
        (adapter/'adapter_config.json').write_text('{"peft_type":"LORA","auto_mapping":null}')
        self.assertEqual(local_adapter_directory(adapter),adapter.resolve())
        (adapter/'adapter_config.json').write_text('{"peft_type":"LORA","auto_mapping":{"custom":1}}')
        with self.assertRaises(ValueError):local_adapter_directory(adapter)

    def test_optional_allowlist_does_not_admit_operational_dependencies(self):
        for filename in ['models/loading.py','models/lora_sft.py']:
            for text in ['import requests','import occlumency.b2_shard','import subprocess']:
                with self.subTest(filename=filename,text=text),self.assertRaises(ValueError):
                    check_imports(text,filename)


if __name__=='__main__':unittest.main()
