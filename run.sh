#!/bin/bash
export HF_TOKEN=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
export PATH=/research/huang/workspaces/hytopot/miniconda3/envs/program1/bin:$PATH
export HF_HOME=/research/huang/workspaces/hytopot/faultdiagnosis/.hf
python3 run.py $@
