from copy import deepcopy
from pathlib import Path
import pytest
from mmengine.config import Config
from fineaction_repro.prepare import native_annotation

ROOT = Path(__file__).resolve().parents[1]


def test_resolved_recipe_retains_official_native_model():
    original=Config.fromfile(str(ROOT/'configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py'))
    current=Config.fromfile(str(ROOT/'configs/adatad/fineaction/e2e_fineaction_videomae_b_768x1_160_adapter.py'))
    assert current.model.backbone==original.model.backbone
    assert current.model.projection==original.model.projection
    assert current.model.rpn_head.num_classes==106
    assert current.model.rpn_head.loss_normalizer==100.0
    assert isinstance(current.model.rpn_head.loss_normalizer,float)
    assert current.scheduler==original.scheduler
    assert current.optimizer==original.optimizer
    assert current.workflow.end_epoch==60
    assert current.solver.train.batch_size==2
    for part in ('train','val','test'):
        assert current.dataset[part].pipeline==original.dataset[part].pipeline
        assert current.dataset[part].block_list is None
    assert current.dataset.train.type=='AnetPaddingDataset'
    assert current.dataset.test.test_mode
    assert current.dataset.test.subset_name=='validation'
    assert current.dataset.test.window_overlap_ratio==0.5
    assert current.evaluation.tiou_thresholds==[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]


def test_native_lengths_do_not_change_ground_truth():
    annotation=dict(database={'a':dict(subset='training',duration=3.4,annotations=[{'segment':[1.,2.],'label':'x'}]),
                              'b':dict(subset='validation',duration=2.,annotations=[]),
                              'c':dict(subset='testing',duration=2.,annotations=[])})
    before=deepcopy(annotation)
    records={'a':dict(status='PASS',frame=87),'b':dict(status='PASS',frame=51)}
    result=native_annotation(annotation,records)
    assert annotation==before
    assert result['database']['a']['frame']==87
    assert result['database']['a']['duration']==3.4
    for name in annotation['database']:
        assert result['database'][name]['annotations']==annotation['database'][name]['annotations']
    assert 'frame' not in result['database']['c']


@pytest.mark.parametrize('records',[{}, {'a':dict(status='FAILED',frame=0)}])
def test_missing_or_failed_media_cannot_create_ready_dataset(records):
    with pytest.raises(ValueError):
        native_annotation(dict(database={'a':dict(subset='training')}),records)
