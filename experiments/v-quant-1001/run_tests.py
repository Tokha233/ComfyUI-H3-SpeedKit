from pathlib import Path
import sys,pytest
r=Path('/tmp/h3-kitchen-followup-1001');sys.path[:0]=[str(r/'candidate'),'/tmp/h3-kitchen-prs-0930/build-deps','/last/deps']
raise SystemExit(pytest.main([str(r/'test_int8_v_quant_schedule.py'),'-q','--tb=short']))
