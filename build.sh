#!/bin/bash
mkdir -p /tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache
pip install -r requirements.txt