import sys,torch
from mmengine.config import Config
sys.path.insert(0,'/data/run01/sczc063/yuzibo/projects/duca_h65_matched_cycle4_55eb7b81_eval2')
from opentad.datasets import build_dataset,build_dataloader
from opentad.models import build_detector
p='/data/run01/sczc063/yuzibo/projects/duca_h65_matched_cycle4_55eb7b81_eval2/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py'
c=Config.fromfile(p)
print('CONFIG_OK',c.get('seed'),c.model.get('single_clock_admission'),c.get('duca_sampling_rate_contract',{}).get('exact_budget'))
ds=build_dataset(c.dataset.train)
print('DATASET_OK',len(ds),type(ds).__name__)
x=ds[0]
print('SAMPLE_KEYS',list(x.keys()) if isinstance(x,dict) else type(x))
print('SAMPLE_SHAPES',{k:(tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in (x.items() if isinstance(x,dict) else [])})
dl=build_dataloader(c.dataset.train)
b=next(iter(dl))
print('BATCH_KEYS',list(b.keys()) if isinstance(b,dict) else type(b))
print('BATCH_SHAPES',{k:(tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in (b.items() if isinstance(b,dict) else [])})
m=build_detector(c.model)
print('MODEL_OK',type(m).__name__,sum(p.numel() for p in m.parameters()))
print('GLOBAL_K384',384)
