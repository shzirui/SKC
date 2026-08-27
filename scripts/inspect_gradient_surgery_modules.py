#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / 'protection/inspect_gradient_surgery_modules.py'), run_name='__main__')
