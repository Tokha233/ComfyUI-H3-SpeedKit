from pathlib import Path
import subprocess
r=Path('/tmp/h3-kitchen-prs-0930')
for pair in range(4):
    for arm in (['base','combined'] if pair%2==0 else ['combined','base']):
        subprocess.run(['python',str(r/'combined-sampler-1001.py'),'--arm',arm,'--pair',str(pair)],check=True)
