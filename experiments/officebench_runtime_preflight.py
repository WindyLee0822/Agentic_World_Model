"""Native runtime smoke test on disposable synthetic files; no benchmark task state."""
import json, os, subprocess, tempfile
from pathlib import Path
from docx import Document
from openpyxl import Workbook
root=Path(tempfile.mkdtemp(prefix='officebench-runtime-proof-'))
for d in ['testbed/data','tmp/home','etc']:(root/d).mkdir(parents=True,exist_ok=True)
(root/'etc/passwd').write_text(f'officebench:x:{os.getuid()}:{os.getgid()}:OfficeBench:/tmp/home:/bin/sh\n')
(root/'etc/group').write_text(f'officebench:x:{os.getgid()}:\n')
(root/'etc/nsswitch.conf').write_text('passwd: files\ngroup: files\n')
d=Document();d.add_paragraph('Conversion proof alpha 7329');d.save(root/'testbed/data/proof.docx')
w=Workbook();w.active['A1']='Spreadsheet proof 8462';w.save(root/'testbed/data/sheet.xlsx')
env=dict(os.environ,HOME='/tmp/home',JAVA_HOME='/usr/lib/jvm/default-java',UNO_JAVA_JFW_ENV_JREHOME='1',LD_LIBRARY_PATH='/usr/lib/libreoffice/program:/usr/lib/jvm/default-java/lib:/usr/lib/jvm/default-java/lib/jli:/usr/lib/jvm/default-java/lib/server')
bindings=[]
for p in ['/usr','/bin','/lib','/lib64','/dev/null','/dev/urandom','/proc/version','/etc/libreoffice','/etc/fonts']:
 if Path(p).exists():bindings += ['-b',p]
for p in Path('/etc').glob('java-*-openjdk'):bindings += ['-b',str(p)]
cmd=['/usr/local/bin/tts-no-network','proot','-r',str(root),*bindings,'-w','/']
report={'root':str(root),'checks':[]}
for operand,needle in [('proof.docx','alpha 7329'),('sheet.xlsx','8462')]:
 result=subprocess.run(cmd+['libreoffice','--headless','--convert-to','pdf','/testbed/data/'+operand,'--outdir','/testbed/data'],env=env,text=True,capture_output=True,timeout=120)
 pdf=root/'testbed/data'/Path(operand).with_suffix('.pdf')
 valid=pdf.exists() and pdf.read_bytes().startswith(b'%PDF-')
 text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True) if valid else ''
 report['checks'].append({'operand':operand,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'valid_pdf':valid,'content_match':needle in text})
result=subprocess.run(cmd+['ls','/testbed/data'],env=env,text=True,capture_output=True,timeout=30)
report['ls']={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
# Network filter remains enforced after adding the statx compatibility fallback.
result=subprocess.run(['/usr/local/bin/tts-no-network','/usr/bin/python3','-c','import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)'],capture_output=True,text=True)
report['network_blocked']=result.returncode!=0 and 'Operation not permitted' in result.stderr
out=Path(os.environ.get('OFFICEBENCH_RUNTIME_REPORT', str(Path(__file__).with_name('runtime_check.json'))));out.parent.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert all(c['valid_pdf'] and c['content_match'] and c['exit_code']==0 and not c['stderr'] for c in report['checks'])
assert report['ls']['exit_code']==0 and report['network_blocked']
