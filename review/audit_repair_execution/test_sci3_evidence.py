"""Test a clean SCI3 revision in a separate CPU-only remote checkout."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = Path('E:/DeskTop/TAD/GeoSparse_SCI3_Evidence_20260908')
sys.path.insert(0, str(HERE))
from deploy_corrected import command

tests = sys.argv[1:] or ['tests/test_sci3_interventions.py', 'tests/test_sci3_bootstrap.py']
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
    raise RuntimeError('SCI3 checkout must be clean before exact-revision tests')
output = HERE / ('sci3_verification_' + sha[:8])
output.mkdir(exist_ok=True)
bundle = output / 'research.bundle'
subprocess.run(['git', 'bundle', 'create', str(bundle), 'b70ae056c495b43ca3f305fe438926b97b2723b5..HEAD'], cwd=REPO, check=True)
root = '/data/run01/sczc063/yuzibo/geosparse_official_20260908/sci3_evidence_' + sha[:8]
remote_bundle = root + '/research.bundle'
upload = 'from pathlib import Path; import sys; p=Path(' + repr(remote_bundle) + '); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())'
result = subprocess.run(command('source', 'python3 -c ' + shlex.quote(upload)), input=bundle.read_bytes(), capture_output=True, timeout=90)
if result.returncode:
    raise RuntimeError(result.stderr.decode(errors='replace'))
script = r'''
from pathlib import Path
import json,os,shutil,subprocess,time
root=Path(ROOT)
b=json.loads(Path('/data/run01/sczc063/yuzibo/geosparse_official_20260908/audit_repair_b70ae056/control/bindings.json').read_text())
base=Path(b['repo_root']); repo=root/'repo'
if not repo.exists():
    subprocess.run(['git','clone','--shared','-q',str(base),str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','-q',str(root/'research.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach','-q',SHA],check=True)
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
for p in base.glob('opentad/**/*.so'):
    q=repo/p.relative_to(base);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
log=root/'pytest.log'
started=time.time()
with log.open('w') as out:
    result=subprocess.run([b['entrypoint'][0],'-m','pytest',*TEST_FILES,'-q'],cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=150)
receipt=dict(source_commit=SHA,repository=str(repo),tests=TEST_FILES,cpu_only=True,synthetic_inputs=True,started_at=started,elapsed_s=time.time()-started,returncode=result.returncode,training_submitted=False)
(root/'receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
print(log.read_text()[-14000:])
raise SystemExit(result.returncode)
'''.replace('ROOT', repr(root)).replace('SHA', repr(sha)).replace('TEST_FILES', repr(tests))
result = subprocess.run(command('source', 'python3 -'), input=script.encode(), capture_output=True, timeout=200)
(output / 'remote.stdout.txt').write_bytes(result.stdout)
(output / 'remote.stderr.txt').write_bytes(result.stderr)
(output / 'receipt.json').write_text(json.dumps(dict(source_commit=sha,remote_root=root,ssh_returncode=result.returncode,completed_at_utc=datetime.now(timezone.utc).isoformat()),indent=2))
print(result.stdout.decode(errors='replace'))
if result.returncode:
    print(result.stderr.decode(errors='replace'))
    raise SystemExit(result.returncode)
