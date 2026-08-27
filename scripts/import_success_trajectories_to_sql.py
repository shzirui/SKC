#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / 'data/import_success_trajectories_to_sql.py'), run_name='__main__')
