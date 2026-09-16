"""Check the descriptive evidence classes, never infer missing training facts."""
from pathlib import Path
import unittest

from analysis.check_package import load

ROOT=Path(__file__).resolve().parents[1]


class ModelConstructionTests(unittest.TestCase):
    def setUp(self):
        self.model=load(ROOT/'metadata/model_construction.json')
        self.experiments=load(ROOT/'metadata/experiment_definitions.json')

    def test_all_98_states_and_precision_tracks(self):
        states=self.model['states']
        self.assertEqual(len(states),98)
        self.assertEqual(len({r['public_state_id'] for r in states}),98)
        self.assertEqual(len({r['track'] for r in states}),10)
        self.assertEqual(len({(r['track'],r['sft_seed'],r['refusal_level']) for r in states}),98)
        for row in states:
            self.assertEqual(row['basis'],'observed_historical')
            self.assertEqual(row['encoding_depth'],8)
            self.assertEqual(row['historical_evaluation_max_new_tokens'],48)

    def test_all_original_training_parameters_are_explicit_unknowns(self):
        block=self.model['historical_training_parameters']
        self.assertEqual(block['basis'],'unknown')
        self.assertTrue(block['reason'])
        self.assertTrue(all(value is None for value in block['values'].values()))
        self.assertIn('optimizer_steps',block['values'])
        self.assertIn('training_precision',block['values'])
        defaults=self.model['reference_training_defaults']
        self.assertEqual(defaults['basis'],'current_code_default')
        self.assertIs(defaults['historically_verified'],False)
        self.assertEqual(defaults['values']['lora_r'],16)
        self.assertEqual(defaults['values']['max_length'],512)

    def test_code_derived_sizes_are_not_observed_counts(self):
        combined=set()
        for row in self.model['states']:
            expected=row['expected_training_rows']
            n,l,d=row['n_canaries'],row['refusal_level'],row['encoding_depth']
            self.assertEqual(expected['injection'],2*d*n)
            self.assertEqual(expected['refusal'],l*n)
            self.assertEqual(expected['combined'],n*(2*d+l))
            self.assertEqual(expected['basis'],'code_derived')
            self.assertIs(expected['counted_from_raw_files'],False)
            combined.add(expected['combined'])
        self.assertEqual(combined,{128,136,144,152,160,256,272,288,304,320,512,544,576,608,640})

    def test_30b_observed_levels_not_replaced_by_launcher_defaults(self):
        self.assertEqual({r['refusal_level'] for r in self.model['states'] if r['track']=='Qwen3-30B-A3B-bf16'},set(range(5)))
        epochs=self.model['launcher_epoch_declarations']
        self.assertEqual(epochs['basis'],'launcher_declared')
        self.assertIs(epochs['historically_verified'],False)
        self.assertEqual(epochs['by_model']['Qwen3-30B-A3B'],5)

    def test_sidecar_hashes_are_not_claimed_to_be_contents(self):
        bindings=[r for t in self.model['tracks'] for r in t['adapter_metadata_bindings']]
        self.assertEqual(len(bindings),59)
        for track in self.model['tracks']:
            self.assertIsNone(track['model_revision'])
        for row in bindings:
            self.assertIs(row['metadata_contents_available'],False)
            self.assertEqual(len(row['train_metadata_sha256']),64)
            self.assertEqual(len(row['adapter_config_sha256']),64)

    def test_descriptor_is_not_runtime_configuration(self):
        self.assertIs(self.experiments['descriptive_only'],True)
        self.assertIs(self.experiments['runtime_configuration'],False)
        self.assertEqual(set(self.experiments['experiments']),{'A1','A2','A3','B1','B2','B3','B4'})
        self.assertEqual(self.experiments['shared_target_decoding']['B1_B2_B3_B4']['max_new_tokens'],256)
        self.assertEqual(self.experiments['shared_target_decoding']['historical_E1_max_new_tokens'],48)
        for name in ['B1','B2','B3']:
            self.assertIn('information_scope',self.experiments['experiments'][name])

    def test_b4_structural_metadata_only(self):
        domain=self.experiments['b4_domain_structure']
        self.assertEqual(domain['factor_cardinalities'],[8,8,8,8])
        self.assertEqual(domain['combination_count_per_canary'],4096)
        self.assertEqual(domain['batches_per_cell'],512)
        self.assertIs(domain['literal_domain_included'],False)
        self.assertIs(domain['enumerator_included'],False)
        self.assertIs(domain['independently_revalidated_during_this_packaging'],False)
        self.assertNotIn('fillers',domain)
        self.assertNotIn('prompts',domain)

    def test_b1_static_information_scope_is_not_an_execution_claim(self):
        scope=self.experiments['experiments']['B1']['initialization_and_screening']
        self.assertEqual(scope['basis'],'static_inspection_of_recorded_runner_version')
        self.assertIs(scope['original_campaign_execution_independently_reverified'],False)
        self.assertEqual(scope['seed_template_family'],'clean')
        self.assertIs(scope['seed_text_interpolates_secret'],False)
        self.assertIs(scope['secret_supplied_to_mutation_or_crossover'],False)
        self.assertEqual(set(scope['secret_used_for']),
                         {'seed_manifest_admission_checks','leakage_adjudication','fitness'})
        self.assertIs(scope['frozen_seed_rows_screened'],True)
        self.assertIs(scope['mutation_filled_initial_candidates_rescreened'],False)
        self.assertIs(scope['later_generated_candidates_rescreened'],False)
        self.assertIs(scope['literal_templates_or_operational_implementation_included'],False)

    def test_matched_analysis_is_label_only(self):
        scope=self.experiments['matched_record_analysis']
        self.assertEqual(scope['reported_roster_size'],66)
        self.assertEqual(scope['reported_eligible_pairs'],30)
        self.assertIs(scope['post_hoc'],True)
        self.assertIs(scope['record_labels_only'],True)
        self.assertIs(scope['source_identity_eligibility_independently_reverified'],False)
        self.assertIs(scope['same_executable_established'],False)
        self.assertIs(scope['new_generation_or_adjudication'],False)

    def test_fresh_joint_sft_is_not_sequential_training(self):
        block=self.model['current_training_behavior']
        self.assertIs(block['base_reset_each_level'],True)
        self.assertIs(block['joint_concatenated_sft'],True)
        self.assertIs(block['separate_injection_then_refusal_optimization_stages'],False)
        self.assertIs(block['continued_level_to_level_adapter_training'],False)


if __name__=='__main__':unittest.main()
