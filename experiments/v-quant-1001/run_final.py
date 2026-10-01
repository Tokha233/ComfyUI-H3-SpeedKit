from pathlib import Path
import subprocess
r=Path('/tmp/h3-kitchen-followup-1001')
for script in ['build_extensions.py','run_tests.py','bench_extensions.py']:
 with (r/(script+'.log')).open('w') as log:subprocess.run(['python',str(r/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
for pair in range(2):
 for arm in (['base','candidate'] if pair==0 else ['candidate','base']):
  with (r/f'sampler-{arm}-{pair}.log').open('w') as log:subprocess.run(['python',str(r/'sampler.py'),'--arm',arm,'--pair',str(pair)],stdout=log,stderr=subprocess.STDOUT,check=True)
(r/'final-success.txt').write_text('success')
