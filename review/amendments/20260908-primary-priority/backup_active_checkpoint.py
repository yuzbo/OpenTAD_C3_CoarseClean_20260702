"""Copy the current N16 checkpoint to existing A100 storage without deleting it."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from deploy_corrected import ssh
from transfer_assets import command

HERE = Path(__file__).resolve().parent
SOURCE = '/data/run01/sczc063/yuzibo/geosparse_official_20260908/tooling_902fa05b/runs/902fa05b/runs/tr-c43d3e9cad58/checkpoint/last.pth'
DESTINATION = '/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/n16_backup/tr-c43d3e9cad58'


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = DESTINATION + '/last_' + stamp + '.pth'
    reader = 'import sys;f=open(' + repr(SOURCE) + ',"rb");import shutil;shutil.copyfileobj(f,sys.stdout.buffer,1024*1024)'
    writer = 'import sys,shutil;from pathlib import Path;p=Path(' + repr(target) + ');p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".partial");f=t.open("wb");shutil.copyfileobj(sys.stdin.buffer,f,1024*1024);f.close();t.replace(p)'
    with (HERE / ('checkpoint_backup_source_' + stamp + '.log')).open('wb') as source_log, (HERE / ('checkpoint_backup_target_' + stamp + '.log')).open('wb') as target_log:
        source = subprocess.Popen(command('source', 'python3 -c ' + shlex.quote(reader)), stdout=subprocess.PIPE, stderr=source_log)
        destination = subprocess.Popen(command('destination', 'python3 -c ' + shlex.quote(writer)), stdin=source.stdout, stdout=subprocess.DEVNULL, stderr=target_log)
        source.stdout.close()
        destination_code = destination.wait()
        source_code = source.wait()
    if source_code or destination_code:
        raise RuntimeError(f'checkpoint copy failed: source={source_code}, destination={destination_code}; both original and logs retained')
    verify = '''
from pathlib import Path
import json,torch
p=Path(TARGET);x=torch.load(p,map_location='cpu')
assert x['provenance']['source_commit']=='902fa05b5c64452ff1c94b82cabce801943d3484'
assert all(k in x for k in ['optimizer','scheduler','grad_scaler','rng','state_dict_ema'])
print(json.dumps(dict(status='BACKUP_VERIFIED',path=str(p),bytes=p.stat().st_size,epoch=x['epoch'],successful_optimizer_updates=x['successful_optimizer_updates'],source_commit=x['provenance']['source_commit'],original_retained=True)))
'''.replace('TARGET', repr(target))
    python = '/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/envs/opentad/bin/python'
    result = json.loads(ssh('destination', python + ' -', verify.encode()))
    result.update(source_path=SOURCE, copied_at_utc=stamp)
    (HERE / 'active_checkpoint_backup.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
