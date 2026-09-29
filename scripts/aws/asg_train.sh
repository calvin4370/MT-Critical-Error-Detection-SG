#!/usr/bin/env bash
# Starts the 16-bit LoRA training job through an EC2 Auto Scaling group, so AWS itself keeps
# retrying the launch until GPU capacity appears (no machine of ours has to stay on). When
# the job ends, the instance removes itself from the group, so no replacement starts; a
# scheduled action scales the group to zero at a fixed time as a hard stop.
#
# Usage: bash scripts/aws/asg_train.sh STOP_TIME_UTC   (e.g. 2026-09-29T13:00:00Z)
# Needs: the AWS CLI signed in, the bucket and the linguasg-ec2 role (see launch.sh).

set -euo pipefail
cd "$(dirname "$0")/../.."
STOP_AT=$1
BUCKET=linguasg-319014022291-apse1
REGION=ap-southeast-2   # Sydney: the nearest region with 24 GB GPUs
NAME=linguasg-train
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS="aws --region ap-southeast-1"

# The instance may remove itself from this one group, and nothing more
cat > /tmp/linguasg-asg-policy.json <<EOF
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": "autoscaling:TerminateInstanceInAutoScalingGroup",
   "Resource": "arn:aws:autoscaling:$REGION:$ACCOUNT:autoScalingGroup:*:autoScalingGroupName/$NAME"}]}
EOF
aws iam put-role-policy --role-name linguasg-ec2 --policy-name linguasg-leave-group \
    --policy-document file:///tmp/linguasg-asg-policy.json

# Latest code and inputs (secrets, private notes and outputs are never bundled)
tar czf /tmp/linguasg-code.tar.gz src configs scripts pyproject.toml uv.lock .python-version README.md LICENSE Dockerfile .dockerignore
$AWS s3 cp /tmp/linguasg-code.tar.gz "s3://$BUCKET/bundle/code.tar.gz" --quiet
$AWS s3 sync data/altered "s3://$BUCKET/data/altered" --exclude "*" --include "train.jsonl" --include "validation.jsonl" --include "test.jsonl" --quiet
$AWS s3 cp data/processed/test.jsonl "s3://$BUCKET/data/processed/test.jsonl" --quiet
$AWS s3 sync data/raw/wmt21_ced "s3://$BUCKET/data/raw/wmt21_ced" --quiet
$AWS s3 sync data/raw/indicmt "s3://$BUCKET/data/raw/indicmt" --quiet

# Start-up script: run the job, then leave the group (which terminates this instance and
# lowers the group's target to zero, so nothing replaces it). This runs however the job ends.
cat > /tmp/linguasg-asg-user-data.sh <<USERDATA
#!/bin/bash
exec > /var/log/linguasg.log 2>&1
leave() {
    TOKEN=\$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
    ID=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
    aws --region $REGION autoscaling terminate-instance-in-auto-scaling-group \\
        --instance-id "\$ID" --should-decrement-desired-capacity || shutdown -h now
}
trap leave EXIT
export BUCKET=$BUCKET
mkdir -p /opt/linguasg && cd /opt/linguasg
aws --region ap-southeast-1 s3 cp s3://$BUCKET/bundle/code.tar.gz . && tar xzf code.tar.gz
bash scripts/aws/ec2_job.sh train
USERDATA

AMI=$(aws ec2 describe-images --region $REGION --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
USER_DATA=$(base64 -w0 /tmp/linguasg-asg-user-data.sh)
cat > /tmp/linguasg-template.json <<EOF
{"ImageId": "$AMI",
 "IamInstanceProfile": {"Name": "linguasg-ec2"},
 "MetadataOptions": {"HttpTokens": "required"},
 "BlockDeviceMappings": [{"DeviceName": "/dev/sda1",
   "Ebs": {"VolumeSize": 150, "VolumeType": "gp3", "Iops": 6000, "Throughput": 500, "DeleteOnTermination": true}}],
 "UserData": "$USER_DATA",
 "TagSpecifications": [{"ResourceType": "instance",
   "Tags": [{"Key": "Name", "Value": "$NAME"}, {"Key": "project", "Value": "linguasg"}]}]}
EOF
aws ec2 create-launch-template --region $REGION --launch-template-name $NAME \
    --launch-template-data file:///tmp/linguasg-template.json --query 'LaunchTemplate.LaunchTemplateId' --output text

# One instance; any of four GPU types (L4 or A10G, small or large machine), in priority order
SUBNETS=$(aws ec2 describe-subnets --region $REGION --filters Name=default-for-az,Values=true \
    --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
cat > /tmp/linguasg-mixed.json <<EOF
{"LaunchTemplate": {"LaunchTemplateSpecification": {"LaunchTemplateName": "$NAME", "Version": "\$Latest"},
   "Overrides": [{"InstanceType": "g6.xlarge"}, {"InstanceType": "g5.xlarge"},
                 {"InstanceType": "g6.2xlarge"}, {"InstanceType": "g5.2xlarge"}]},
 "InstancesDistribution": {"OnDemandAllocationStrategy": "prioritized", "OnDemandBaseCapacity": 1,
   "OnDemandPercentageAboveBaseCapacity": 100}}
EOF
aws autoscaling create-auto-scaling-group --region $REGION --auto-scaling-group-name $NAME \
    --mixed-instances-policy file:///tmp/linguasg-mixed.json \
    --min-size 0 --max-size 1 --desired-capacity 1 --vpc-zone-identifier "$SUBNETS" \
    --tags "Key=project,Value=linguasg,PropagateAtLaunch=true"

# Hard stop: whatever happens, the group goes to zero at this time
aws autoscaling put-scheduled-update-group-action --region $REGION --auto-scaling-group-name $NAME \
    --scheduled-action-name hard-stop --start-time "$STOP_AT" --min-size 0 --max-size 0 --desired-capacity 0
echo "Auto Scaling group $NAME created; hard stop at $STOP_AT"
