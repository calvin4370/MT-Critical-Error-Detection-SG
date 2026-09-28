#!/usr/bin/env bash
# Uploads the code and inputs to S3, then starts one EC2 GPU instance that runs
# scripts/aws/ec2_job.sh and terminates itself (also after a hard time limit).
#
# Usage: bash scripts/aws/launch.sh eval|train
# Needs: the AWS CLI signed in (aws login), the bucket and the linguasg-ec2 role.

set -euo pipefail
cd "$(dirname "$0")/../.."
ROLE=$1
BUCKET=linguasg-319014022291-apse1
REGION=ap-southeast-2   # Sydney: the nearest region with 24 GB GPUs (Singapore has 16 GB T4s only)
MINUTES=300; [ "$ROLE" = train ] && MINUTES=720
S3="s3://$BUCKET"
AWS="aws --region ap-southeast-1"

# Code only: secrets (.env), private notes, data and outputs are never bundled
tar czf /tmp/linguasg-code.tar.gz src configs scripts pyproject.toml uv.lock .python-version README.md LICENSE Dockerfile .dockerignore
$AWS s3 cp /tmp/linguasg-code.tar.gz "$S3/bundle/code.tar.gz" --quiet
$AWS s3 sync data/altered "$S3/data/altered" --exclude "*" --include "train.jsonl" --include "validation.jsonl" --include "test.jsonl" --quiet
$AWS s3 cp data/processed/test.jsonl "$S3/data/processed/test.jsonl" --quiet
$AWS s3 sync data/raw/wmt21_ced "$S3/data/raw/wmt21_ced" --quiet
$AWS s3 sync data/raw/indicmt "$S3/data/raw/indicmt" --quiet
$AWS s3 sync outputs/finetune/evaluator/adapter "$S3/adapters/evaluator-qlora" --quiet
$AWS s3 cp outputs/harness/sealion/translations.jsonl "$S3/inputs/outputs/harness/sealion/translations.jsonl" --quiet

# The instance's start-up script: fetch the code, run the job, then shut down (= terminate)
cat > /tmp/linguasg-user-data.sh <<USERDATA
#!/bin/bash
exec > /var/log/linguasg.log 2>&1
shutdown -h +$MINUTES
export BUCKET=$BUCKET
mkdir -p /opt/linguasg && cd /opt/linguasg
aws --region ap-southeast-1 s3 cp s3://$BUCKET/bundle/code.tar.gz . && tar xzf code.tar.gz
bash scripts/aws/ec2_job.sh $ROLE
shutdown -h now
USERDATA

AMI=$(aws ec2 describe-images --region $REGION --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
# GPU capacity varies by zone and type: try each zone, first the L4 (g6), then the A10G (g5),
# then the same GPUs on larger machines (8 vCPUs, more RAM)
SUBNETS=$(aws ec2 describe-subnets --region $REGION --filters Name=default-for-az,Values=true \
    --query 'Subnets[].SubnetId' --output text)
for TYPE in g6.xlarge g5.xlarge g6.2xlarge g5.2xlarge; do
    for SUBNET in $SUBNETS; do
        if aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type $TYPE --count 1 \
            --subnet-id "$SUBNET" --iam-instance-profile Name=linguasg-ec2 \
            --instance-initiated-shutdown-behavior terminate \
            --metadata-options HttpTokens=required \
            --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,Iops=6000,Throughput=500,DeleteOnTermination=true}' \
            --user-data file:///tmp/linguasg-user-data.sh \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=linguasg-$ROLE},{Key=project,Value=linguasg}]" \
            --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]' --output text; then
            exit 0
        fi
        echo "no $TYPE capacity in $SUBNET, trying the next"
    done
done
echo "no GPU capacity in $REGION"
exit 1
