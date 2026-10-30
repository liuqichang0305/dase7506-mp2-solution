"""Generated saved-output fixtures and mocked diagnostics; no real holdout I/O."""
import copy
from contextlib import contextmanager, ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('selected_confidence_under_test', REPO/'code/tools/prepare_selected_confidence_v3.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class SelectedConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mp2-v3-selected-confidence-synthetic-')
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root/'results/research_v3/selected_confidence'
        self.access = {'validation_open':False, 'rejected':[]}
        self.locked = False
        self.validation_calls = 0
        self.selected = {'name':'synthetic_selected', 'historical_fallback':True,
            'selector':{'kind':'single_model','experiment_id':'synthetic_selected','family':'deepsets'}}
        self.summary = {'synthetic':'completed validation identity'}
        self.verified = {'lock_sha256':'a'*64}
        self.assets = {'artifacts/research_v3/final_selected/config.json':'b'*64}
        self.sources = {'code/synthetic.py':'c'*64}
        self.metrics = {'n':798, 'correct':700}
        self.selected_path = self.root/'submission/v3/SELECTED_MODEL.json'
        self.selected_path.parent.mkdir(parents=True)
        self.selected_path.write_text(json.dumps(self.selected),encoding='utf-8')
        for generation in ('research_v2','research_v3'):
            folder=self.root/'results'/generation/'validation'/self.selected['name']
            folder.mkdir(parents=True)
            (folder/'validation-predictions.json').write_text(json.dumps({'synthetic':generation,'selected':[0]}),encoding='utf-8')
            (folder/'validation-score.json').write_text(json.dumps({'synthetic':generation,'correct':700,'n':798}),encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    @contextmanager
    def resource_lock(self):
        self.locked=True
        try:yield
        finally:self.locked=False

    def validate_selected(self, root):
        self.assertEqual(root,self.root)
        self.assertTrue(self.locked)
        if not self.validation_calls:self.assertFalse(self.access['validation_open'])
        self.validation_calls+=1
        return copy.deepcopy((self.selected,self.summary,self.verified,self.assets,{'within_limit':True}))

    def diagnostic(self, finalist, verified, record):
        self.assertTrue(self.locked)
        self.assertTrue(self.access['validation_open'])
        self.assertEqual(finalist['checkpoint'],helper.freezing.CHECKPOINT)
        self.assertEqual(finalist['selector'],self.selected['selector'])
        self.assertEqual(helper.confirmation.OUT,self.output)
        self.assertEqual(verified['lock_sha256'],self.verified['lock_sha256'])
        self.assertIn('submission/v3/SELECTED_MODEL.json',verified['lock']['inference_source_sha256'])
        (self.output/'validation'/self.selected['name']/'validation-events.json').write_text('{"synthetic":true}',encoding='utf-8')
        return self.metrics

    def invoke(self, diagnostic=None, validator=None, source_provider=None, lock=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(helper,'ROOT',self.root))
            stack.enter_context(patch.object(helper.confirmation,'install_data_guard',return_value=self.access))
            stack.enter_context(patch.object(helper.confirmation,'exclusive_evaluation',lock or self.resource_lock))
            stack.enter_context(patch.object(helper.freezing,'validate_selected',side_effect=validator or self.validate_selected))
            stack.enter_context(patch.object(helper.freezing,'source_hashes',side_effect=source_provider or (lambda root:dict(self.sources))))
            stack.enter_context(patch.object(helper.confirmation,'diagnostic_events',side_effect=diagnostic or self.diagnostic))
            # This helper must never invoke fresh official prediction/scoring.
            stack.enter_context(patch.object(helper.confirmation,'evaluate_candidate',side_effect=AssertionError('New official evaluation forbidden')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            helper.main()

    def test_historical_fallback_copies_saved_outputs_and_rechecks_identity(self):
        original_out=helper.confirmation.OUT
        self.invoke()
        self.assertEqual(helper.confirmation.OUT,original_out)
        self.assertEqual(self.validation_calls,2)
        record=json.loads((self.output/'preparation.json').read_text(encoding='utf-8'))
        self.assertTrue(record['historical_fallback'])
        self.assertEqual(record['new_official_score_runs'],0)
        self.assertFalse(record['candidate_selection_changed'])
        self.assertTrue(all(name.startswith('results/research_v2/') for name in record['original_official_output_sha256']))
        copied=json.loads((self.output/'validation'/self.selected['name']/'validation-predictions.json').read_text())
        self.assertEqual(copied['synthetic'],'research_v2')

    def test_new_selected_candidate_uses_saved_v3_outputs(self):
        self.selected['historical_fallback']=False
        self.invoke()
        record=json.loads((self.output/'preparation.json').read_text())
        self.assertFalse(record['historical_fallback'])
        self.assertTrue(all(name.startswith('results/research_v3/') for name in record['original_official_output_sha256']))

    def test_freeze_or_test_guard_runs_inside_resource_lock(self):
        for name in ('FREEZE.json','TEST_OPENED.json'):
            @contextmanager
            def lock():
                self.locked=True
                marker=self.root/'submission/v3'/name
                marker.write_text('{}')
                try:yield
                finally:marker.unlink();self.locked=False
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'before final freeze/test'):
                self.invoke(lock=lock)
            self.assertFalse(self.access['validation_open'])
            self.assertEqual(self.validation_calls,0)

    def test_invalid_selection_never_opens_validation(self):
        def reject(root):
            self.assertFalse(self.access['validation_open'])
            raise ValueError('Synthetic unique selection invalid')
        with self.assertRaisesRegex(ValueError,'unique selection'):
            self.invoke(validator=reject)
        self.assertFalse(self.access['validation_open'])
        self.assertFalse(self.output.exists())

    def test_existing_evidence_never_overwritten(self):
        self.output.mkdir(parents=True)
        sentinel=self.output/'sentinel.txt';sentinel.write_text('preserve')
        with self.assertRaises(FileExistsError):self.invoke()
        self.assertEqual(sentinel.read_text(),'preserve')

    def test_changed_choices_preserve_partial_output_and_restore_global(self):
        original_out=helper.confirmation.OUT
        def changed(*args):raise ValueError('Diagnostic inference changed saved official choices')
        with self.assertRaisesRegex(ValueError,'changed saved official choices'):
            self.invoke(diagnostic=changed)
        self.assertEqual(helper.confirmation.OUT,original_out)
        self.assertTrue((self.output/'validation'/self.selected['name']/'validation-predictions.json').exists())
        self.assertFalse((self.output/'preparation.json').exists())

    def test_post_diagnostic_source_or_selected_byte_change_rejected(self):
        def changed(*args):
            result=self.diagnostic(*args)
            self.selected_path.write_text('{"synthetic":"changed identity bytes"}')
            return result
        with self.assertRaisesRegex(ValueError,'changed during analysis'):
            self.invoke(diagnostic=changed)
        self.assertFalse((self.output/'preparation.json').exists())

    def test_post_diagnostic_asset_change_rejected(self):
        def changed(root):
            record=self.validate_selected(root)
            if self.validation_calls==2:record[3]['artifacts/research_v3/final_selected/config.json']='d'*64
            return record
        with self.assertRaisesRegex(ValueError,'changed during analysis'):
            self.invoke(validator=changed)
        self.assertFalse((self.output/'preparation.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
