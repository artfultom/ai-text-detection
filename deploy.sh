#!/bin/bash
set -e

FOLDER_ID=$(yc config get folder-id)
CLOUD_ID=$(yc config get cloud-id)

IMAGE_NAME="ai-text-detector"
REGISTRY_NAME="ai-text-registry"
CLUSTER_NAME="ai-text-cluster"
NODE_GROUP_NAME="ai-text-nodes"

ZONE="ru-central1-a"
SERVICE_ACCOUNT_NAME="k8s-sa"

TAG="latest"

echo "Using folder: $FOLDER_ID"
echo "Cloud: $CLOUD_ID"

echo "Creating container registry (if not exists)..."

REGISTRY_ID=$(yc container registry list --format json | jq -r ".[] | select(.name==\"$REGISTRY_NAME\") | .id")

if [ -z "$REGISTRY_ID" ]; then
  REGISTRY_ID=$(yc container registry create --name "$REGISTRY_NAME" --format json | jq -r .id)
fi

REGISTRY="cr.yandex/$REGISTRY_ID"

echo "Registry: $REGISTRY"

NETWORK_NAME="ai-text-net"
SUBNET_NAME="ai-text-subnet"

echo "Creating VPC network..."

NETWORK_ID=$(yc vpc network list --format json | jq -r ".[] | select(.name==\"$NETWORK_NAME\") | .id")

if [ -z "$NETWORK_ID" ]; then
  NETWORK_ID=$(yc vpc network create --name "$NETWORK_NAME" --format json | jq -r .id)
fi

echo "Network: $NETWORK_ID"

echo "Creating subnet..."

SUBNET_ID=$(yc vpc subnet list --format json | jq -r ".[] | select(.name==\"$SUBNET_NAME\") | .id")

if [ -z "$SUBNET_ID" ]; then
  SUBNET_ID=$(yc vpc subnet create \
    --name "$SUBNET_NAME" \
    --zone "$ZONE" \
    --range 10.10.0.0/24 \
    --network-id "$NETWORK_ID" \
    --format json | jq -r .id)
fi

echo "Subnet: $SUBNET_ID"

echo "Creating service account..."

SA_ID=$(yc iam service-account list --format json | jq -r ".[] | select(.name==\"$SERVICE_ACCOUNT_NAME\") | .id")

if [ -z "$SA_ID" ]; then
  SA_ID=$(yc iam service-account create --name "$SERVICE_ACCOUNT_NAME" --format json | jq -r .id)
fi

yc resource-manager folder add-access-binding "$FOLDER_ID" \
  --role container-registry.images.puller \
  --subject serviceAccount:"$SA_ID" || true

yc resource-manager folder add-access-binding "$FOLDER_ID" \
  --role k8s.clusters.agent \
  --subject serviceAccount:"$SA_ID" || true

yc resource-manager folder add-access-binding "$FOLDER_ID" \
  --role vpc.publicAdmin \
  --subject serviceAccount:"$SA_ID" || true

yc resource-manager folder add-access-binding "$FOLDER_ID" \
  --role iam.serviceAccounts.user \
  --subject serviceAccount:"$SA_ID" || true

yc resource-manager folder add-access-binding "$FOLDER_ID" \
  --role load-balancer.admin \
  --subject serviceAccount:"$SA_ID" || true

echo "Creating Kubernetes cluster..."

CLUSTER_ID=$(yc managed-kubernetes cluster list --format json | jq -r ".[] | select(.name==\"$CLUSTER_NAME\") | .id")

if [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ID=$(yc managed-kubernetes cluster create \
    --name "$CLUSTER_NAME" \
    --network-id "$NETWORK_ID" \
    --zone "$ZONE" \
    --subnet-id "$SUBNET_ID" \
    --public-ip \
    --service-account-id "$SA_ID" \
    --node-service-account-id "$SA_ID" \
    --release-channel regular \
    --format json | jq -r .id)
fi

echo "Cluster: $CLUSTER_ID"

echo "Creating node group..."

NODE_GROUP_ID=$(yc managed-kubernetes node-group list --format json | jq -r ".[] | select(.name==\"$NODE_GROUP_NAME\") | .id")

if [ -z "$NODE_GROUP_ID" ]; then
  yc managed-kubernetes node-group create \
    --cluster-id "$CLUSTER_ID" \
    --name "$NODE_GROUP_NAME" \
    --fixed-size 1 \
    --cores 4 \
    --memory 8 \
    --core-fraction 100 \
    --disk-size 30 \
    --preemptible \
    --platform standard-v3 \
    --network-interface subnets=$SUBNET_ID,ipv4-address=nat
fi

echo "Waiting for Kubernetes API endpoint..."

for i in {1..30}; do
  ENDPOINT=$(yc managed-kubernetes cluster get "$CLUSTER_ID" --format json | jq -r '.master.endpoints.external_v4_endpoint')

  if [ "$ENDPOINT" != "null" ] && [ -n "$ENDPOINT" ]; then
    echo "Endpoint ready: $ENDPOINT"
    break
  fi

  echo "still provisioning... ($i/30)"
  sleep 10
done

echo "Fetching kubeconfig..."

yc managed-kubernetes cluster get-credentials "$CLUSTER_ID" --external --force

echo "Building docker image..."
docker build --platform=linux/amd64 -t $IMAGE_NAME .

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"

echo "Tag: $FULL_IMAGE"
docker tag $IMAGE_NAME $FULL_IMAGE

echo "Push..."
docker push $FULL_IMAGE

echo "Deploying manifests..."

kubectl apply -f k8s/namespace.yaml || true

sed "s|<YOUR_REGISTRY>|$REGISTRY_ID|g" k8s/deployment.yaml | kubectl apply -f -

kubectl apply -f k8s/service.yaml

echo "Waiting for external IP..."
kubectl get svc ai-text-detector-svc -w