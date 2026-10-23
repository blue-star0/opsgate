#!/bin/bash
# EC2 user-data: runs Strands Decider 2B and OpsGate on ONE instance.
# Suggested: g5.xlarge (24 GB GPU, Deep Learning AMI). CPU works too, just slower:
# the Decider README says the 2B model also serves on CPU.
# Attach an instance profile with deploy/iam-policy.json (Bedrock invoke only by default).
set -euxo pipefail
cd /home/ec2-user

python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install strands-decider strands-agents boto3

# 1) Decider 2B served locally (weights come from Hugging Face on first run).
nohup strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099 \
  > decider.log 2>&1 &

# 2) OpsGate (clone your public repo here).
# git clone https://github.com/<you>/opsgate && cd opsgate && pip install -e .
# AWS_REGION=us-east-1 opsgate --decider-url http://127.0.0.1:8099
