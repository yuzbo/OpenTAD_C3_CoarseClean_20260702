"""Launch upstream AdaTAD and retain full-validation EMA checkpoint selection."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fineaction_repro.prepare import save


def resolve(data, weights, output):
    from mmengine.config import Config
    cfg=Config.fromfile(str(ROOT/'configs/adatad/fineaction/e2e_fineaction_videomae_b_768x1_160_adapter.py'))
    for part in ('train','val','test'):
        cfg.dataset[part].ann_file=str(data/'annotations/native_metadata.json')
        cfg.dataset[part].class_map=str(data/'annotations/category_idx.txt')
        cfg.dataset[part].data_path=str(data/'videos')
    cfg.evaluation.ground_truth_filename=str(data/'annotations/annotations_gt.json')
    cfg.model.backbone.custom.pretrain=str(weights)
    cfg.work_dir=str(output)
    return cfg


def datasets(cfg, output):
    from opentad.datasets import build_dataset
    db=json.loads(Path(cfg.dataset.train.ann_file).read_text())['database']
    train=build_dataset(cfg.dataset.train)
    test=build_dataset(cfg.dataset.test)
    expected_train={k for k,v in db.items() if v['subset']=='training'}
    expected_val={k for k,v in db.items() if v['subset']=='validation'}
    actual_train={v[0] for v in train.data_list}
    actual_val={v[0] for v in test.data_list}
    assert len(expected_train)==8440 and len(expected_val)==4174
    assert actual_val==expected_val and actual_train<=expected_train
    report=dict(training_pool=len(expected_train),loaded_training_videos=len(train),
                filtered_invalid_gt_videos=sorted(expected_train-actual_train),
                validation_videos=len(actual_val),validation_windows=len(test),
                internal_holdout=0,filter_policy='upstream AnetPaddingDataset filter_gt=True, absolute valid intervals')
    save(output/'data_contract.json',report)
    return train,test,expected_val


def gpu_precheck(cfg, train, test, output):
    import logging
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from opentad.datasets import build_dataloader
    from opentad.models import build_detector
    from opentad.cores import build_optimizer
    from opentad.utils import set_seed
    set_seed(0,False)
    dist.init_process_group('nccl')
    torch.cuda.set_device(0)
    model=build_detector(cfg.model).cuda()
    model=DistributedDataParallel(model,device_ids=[0],static_graph=True)
    optimizer=build_optimizer(deepcopy(cfg.optimizer),model,logging.getLogger('precheck'))
    loader=build_dataloader(train,2,0,1,shuffle=False,num_workers=0)
    data=next(iter(loader))
    assert tuple(data['inputs'].shape)==(2,1,3,768,160,160)
    model.train();optimizer.zero_grad()
    scaler=torch.cuda.amp.GradScaler(init_scale=128.)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        losses=model(**data,return_loss=True)
    assert all(torch.isfinite(value).all() for value in losses.values())
    scaler.scale(losses['cost']).backward();scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    assert torch.isfinite(norm) and norm>0
    scaler.step(optimizer);scaler.update()
    model.eval()
    sample=next(iter(build_dataloader(test,1,0,1,shuffle=False,num_workers=0)))
    post=deepcopy(cfg.post_processing);post.sliding_window=True
    with torch.no_grad(),torch.cuda.amp.autocast(dtype=torch.float16):
        predictions=model(**sample,return_loss=False,infer_cfg=cfg.inference,
                          post_cfg=post,ext_cls=test.class_map)
    from opentad.cores.test_engine import gather_ddp_results
    predictions=gather_ddp_results(1,predictions,post)
    assert set(predictions)=={sample['metas'][0]['video_name']}
    save(output/'gpu_precheck.json',dict(status='PASS',input_shape=list(data['inputs'].shape),
         losses={k:float(v) for k,v in losses.items()},gradient_norm=float(norm),
         optimizer_step=True,prediction_videos=list(predictions),
         gpu=torch.cuda.get_device_name(0),peak_vram_bytes=torch.cuda.max_memory_allocated(),
         scope='real two-video training forward/backward/update and one-window inference; not mAP',
         slurm_id=os.environ.get('SLURM_JOB_ID')))
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['config','precheck','train'],required=True)
    args=p.parse_args()
    data,weights,output=map(Path,(args.data,args.weights,args.output))
    output.mkdir(parents=True,exist_ok=True)
    assert json.loads((data/'receipt.json').read_text())['status']=='PASS'
    assert weights.is_file()
    cfg=resolve(data,weights,output)
    config=output/'resolved_config.py'
    if config.exists():
        from mmengine.config import Config
        assert Config.fromfile(str(config)).to_dict()==cfg.to_dict(), 'configuration changed within this run'
    cfg.dump(str(config))
    train,test,names=datasets(cfg,output)
    if args.mode=='config': return
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work needs a compute allocation'
    if args.mode=='precheck':
        gpu_precheck(cfg,train,test,output);return
    assert json.loads((output/'gpu_precheck.json').read_text())['status']=='PASS'
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    run=output/'gpu1_id0'
    expected_epochs=list(range(4,60,5))

    def capture(epoch):
        from opentad.evaluations import build_evaluator
        prediction=run/'result_detection.json'
        values=json.loads(prediction.read_text())
        assert set(values['results'])==names, 'incomplete full-validation prediction coverage'
        evaluator=build_evaluator(dict(prediction_filename=values,**cfg.evaluation))
        metrics={k:float(v) for k,v in evaluator.evaluate().items()}
        checkpoint=run/'checkpoint'/f'epoch_{epoch}.pth'
        assert checkpoint.is_file()
        folder=output/f'epoch_{epoch:03d}';folder.mkdir(exist_ok=True)
        shutil.copyfile(prediction,folder/'predictions.json')
        row=dict(metrics=metrics,checkpoint_epoch=epoch,completed_epochs=epoch+1,weights='ema',
                 validation_videos=len(names),validation_windows=len(test),seed=0,source_commit=source,
                 checkpoint_path=str(checkpoint),is_mock=False,protocol='ADATAD_FINEACTION_ADAPTATION')
        save(folder/'metrics.json',row)
        previous=json.loads((output/'best.json').read_text()) if (output/'best.json').exists() else None
        score=metrics['average_mAP']
        if previous is None or score>previous['metrics']['average_mAP'] or (
            score==previous['metrics']['average_mAP'] and epoch<previous['checkpoint_epoch']):
            shutil.copyfile(checkpoint,output/'best.pth');save(output/'best.json',row)

    def launch(mode,checkpoint=None,fixed_epoch=None):
        argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
              str(ROOT/'tools'/('train.py' if mode=='train' else 'test.py')),str(config),'--seed','0','--id','0']
        if checkpoint:
            argv += ['--resume' if mode=='train' else '--checkpoint',str(checkpoint)]
        with (output/'allocations.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),source_commit=source,mode=mode,
                slurm_id=os.environ['SLURM_JOB_ID'],command=argv))+'\n')
        epoch=fixed_epoch
        process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT)),
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        with (output/'console.log').open('a',buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                if 'INFO:' in line or 'Error' in line or 'Traceback' in line: print(line,end='',flush=True)
                match=re.search(r'\[Train\]: Epoch (\d+) started',line)
                if match:
                    epoch=int(match[1]);save(output/'progress.json',dict(status='TRAINING',active_epoch=epoch,
                        completed_epochs=epoch,slurm_id=os.environ['SLURM_JOB_ID']))
                if 'Average-mAP:' in line:
                    assert epoch in expected_epochs
                    capture(epoch)
        code=process.wait()
        if code:
            save(output/'failure.json',dict(status='FAILED',mode=mode,returncode=code,time=time.time()))
            raise RuntimeError(f'upstream {mode} exited {code}; see console.log')

    checkpoints={int(p.stem.split('_')[-1]):p for p in (run/'checkpoint').glob('epoch_*.pth')}
    latest=max(checkpoints,default=-1)
    # Saved epochs whose evaluation was interrupted are backfilled without retraining.
    for epoch in expected_epochs:
        if epoch<=latest and not (output/f'epoch_{epoch:03d}/metrics.json').exists():
            assert epoch in checkpoints, f'missing required epoch{epoch} checkpoint'
            launch('evaluate',checkpoints[epoch],epoch)
    if latest<59: launch('train',checkpoints.get(latest))
    complete=all((output/f'epoch_{e:03d}/metrics.json').is_file() for e in expected_epochs)
    trained=(run/'checkpoint/epoch_59.pth').is_file()
    result=dict(status='COMPLETED' if trained and complete else 'INCOMPLETE',training_complete=trained,
                selection_complete=complete,completed_epochs=60 if trained else None,
                source_commit=source,protocol='ADATAD_FINEACTION_ADAPTATION',seed=0,is_mock=False)
    save(output/'result.json',result)
    if not trained or not complete: raise RuntimeError('training or full selection incomplete')


if __name__=='__main__':
    main()
