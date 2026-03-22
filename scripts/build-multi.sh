#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "错误：必须提供 TAG 参数"
  echo "用法：./build-multi.sh <TAG>"
  exit 1
fi

TAG=$1

REPO="cosincox/easyvoice"

# 创建或使用现有的多架构构建器
if ! docker buildx inspect multiarch-builder >/dev/null 2>&1; then
  sudo docker buildx create --name multiarch-builder --driver docker-container --use
else
  sudo docker buildx use multiarch-builder
fi
sudo docker buildx inspect --bootstrap

sudo docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t "${REPO}:${TAG}" \
  -t "${REPO}:latest" \
  --push \
  .

# sudo docker buildx rm multiarch-builder

echo "完成！多架构镜像已构建并推送为 ${REPO}:${TAG} 和 ${REPO}:latest"