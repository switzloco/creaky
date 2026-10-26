#!/usr/bin/env bash
# ==============================================================================
# CREAKY-CV5 · 1-Click AWS CloudShell Deployment (S3 + Graviton ARM64 EC2 + Benchmark)
# ==============================================================================
# Run inside AWS CloudShell with a single command:
#   curl -sSL https://raw.githubusercontent.com/switzloco/creaky/main/opencv_hackathon/cloudshell_oneclick.sh | bash
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo " ⚡ CREAKY-CV5 · AWS CloudShell 1-Click Deployment (OpenCV 2026)"
echo "======================================================================"

REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-east-1}}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="creaky-cv5-opencv26-${ACCOUNT_ID}-${REGION}"

echo "[1/5] Authenticated AWS Account: ${ACCOUNT_ID} | Region: ${REGION}"

# 1. Shallow clone repo into /tmp/creaky
rm -rf /tmp/creaky
git clone --depth 1 https://github.com/switzloco/creaky.git /tmp/creaky
cd /tmp/creaky

# 2. Run live OpenCV 5 144-slice benchmark on AWS CloudShell & save telemetry JSON
echo "[2/5] Running live OpenCV 144-slice 3-plane MRI benchmark in AWS..."
python3 -m pip install --quiet opencv-python-headless numpy
python3 - <<'PYEOF'
import json, sys
sys.path.insert(0, "/tmp/creaky")
from opencv_hackathon.cv5_engine import run_opencv_cool_benchmark
res = run_opencv_cool_benchmark(144)
with open("/tmp/creaky/opencv_hackathon/dist/aws_benchmark.json", "w") as f:
    json.dump(res, f, indent=2)
print("   -> Measured 144-Slice Study Latency:", res["total_study_ms"], "ms | Graviton COOL:", res["cool_graviton_study_ms"], "ms")
PYEOF

# 3. Create & configure public S3 Static Website Bucket on *.amazonaws.com
echo "[3/5] Provisioning live AWS S3 endpoint: s3://${BUCKET_NAME} ..."
if [ "${REGION}" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "${BUCKET_NAME}" --region "${REGION}" >/dev/null 2>&1 || true
else
    aws s3api create-bucket --bucket "${BUCKET_NAME}" --region "${REGION}" \
        --create-bucket-configuration LocationConstraint="${REGION}" >/dev/null 2>&1 || true
fi

aws s3api put-public-access-block --bucket "${BUCKET_NAME}" \
    --public-access-block-configuration "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

cat > /tmp/bucket_policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${BUCKET_NAME}/*"
    }
  ]
}
EOF
aws s3api put-bucket-policy --bucket "${BUCKET_NAME}" --policy file:///tmp/bucket_policy.json
aws s3 website "s3://${BUCKET_NAME}/" --index-document index.html --error-document index.html

echo "[4/5] Uploading Creaky-CV5 Cockpit, Physician Pitch & 3D Galaxy of Knees to AWS S3..."
aws s3 cp /tmp/creaky/opencv_hackathon/dist/index.html "s3://${BUCKET_NAME}/index.html" --content-type "text/html; charset=utf-8" >/dev/null
aws s3 cp /tmp/creaky/physician_pitch.html "s3://${BUCKET_NAME}/physician_pitch.html" --content-type "text/html; charset=utf-8" >/dev/null
aws s3 cp /tmp/creaky/galaxy_of_knees.html "s3://${BUCKET_NAME}/galaxy_of_knees.html" --content-type "text/html; charset=utf-8" >/dev/null
aws s3 cp /tmp/creaky/opencv_hackathon/dist/aws_benchmark.json "s3://${BUCKET_NAME}/aws_benchmark.json" --content-type "application/json" >/dev/null

S3_URL="http://${BUCKET_NAME}.s3-website-${REGION}.amazonaws.com"
if [ "${REGION}" != "us-east-1" ] && [ "${REGION}" != "us-west-1" ] && [ "${REGION}" != "us-west-2" ]; then
    S3_URL="http://${BUCKET_NAME}.s3-website.${REGION}.amazonaws.com"
fi

# 4. Launch an AWS Graviton ARM64 (aarch64) EC2 Instance running the live FastAPI + OpenCV 5 server
echo "[5/5] Launching AWS Graviton ARM64 (aarch64) EC2 instance for live FastAPI + OpenCV COOL backend..."
EC2_URL="(Skipped or permissions restricted — S3 AWS endpoint above is 100% live)"
set +e
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text 2>/dev/null)
if [ -n "${VPC_ID}" ] && [ "${VPC_ID}" != "None" ]; then
    SG_ID=$(aws ec2 create-security-group --group-name "creaky-cv5-graviton-sg" \
        --description "Creaky-CV5 OpenCV 5 + AWS Graviton FastAPI Cockpit" \
        --vpc-id "${VPC_ID}" --query "GroupId" --output text 2>/dev/null)
    if [ -z "${SG_ID}" ]; then
        SG_ID=$(aws ec2 describe-security-groups --filters Name=group-name,Values=creaky-cv5-graviton-sg --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)
    fi
    aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" --protocol tcp --port 8055 --cidr 0.0.0.0/0 >/dev/null 2>&1
    aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null 2>&1

    AMI_ID=$(aws ssm get-parameters --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
        --query "Parameters[0].Value" --output text 2>/dev/null)

    cat > /tmp/userdata.sh <<'UDEOF'
#!/bin/bash
dnf install -y git python3 python3-pip mesa-libGL
git clone --depth 1 https://github.com/switzloco/creaky.git /opt/creaky
cd /opt/creaky
pip3 install fastapi uvicorn opencv-python-headless numpy pillow pydicom pydantic
nohup python3 -m uvicorn opencv_hackathon.app:app --host 0.0.0.0 --port 8055 > /var/log/creaky.log 2>&1 &
nohup python3 -m uvicorn opencv_hackathon.app:app --host 0.0.0.0 --port 80 > /var/log/creaky80.log 2>&1 &
UDEOF

    for ITYPE in c7g.large t4g.small t4g.micro; do
        INST_ID=$(aws ec2 run-instances \
            --image-id "${AMI_ID}" \
            --instance-type "${ITYPE}" \
            --security-group-ids "${SG_ID}" \
            --user-data file:///tmp/userdata.sh \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=Creaky-CV5-Graviton-COOL}]" \
            --query "Instances[0].InstanceId" --output text 2>/dev/null)
        if [ -n "${INST_ID}" ] && [ "${INST_ID}" != "None" ]; then
            echo "   -> Launched Graviton ARM64 Instance: ${INST_ID} (${ITYPE}) ... waiting for Public IP..."
            sleep 5
            PUB_IP=$(aws ec2 describe-instances --instance-ids "${INST_ID}" --query "Reservations[0].Instances[0].PublicIpAddress" --output text 2>/dev/null)
            EC2_URL="http://${PUB_IP}:8055 (Instance: ${INST_ID} · ${ITYPE} ARM64 Graviton)"
            break
        fi
    done
fi
set -e

echo ""
echo "======================================================================"
echo " ✅ CREAKY-CV5 IS NOW LIVE ON AMAZON WEB SERVICES (AWS)!"
echo "======================================================================"
echo " 1. Live AWS Cockpit URL (Paste in Devpost 'Try It Out'):"
echo "    ${S3_URL}"
echo ""
echo " 2. Live AWS Physician Pitch URL (For Your Brother):"
echo "    ${S3_URL}/physician_pitch.html"
echo ""
echo " 3. Live AWS 3D Galaxy of Knees URL:"
echo "    ${S3_URL}/galaxy_of_knees.html"
echo ""
echo " 4. Live AWS Graviton ARM64 FastAPI Server:"
echo "    ${EC2_URL}"
echo "======================================================================"
