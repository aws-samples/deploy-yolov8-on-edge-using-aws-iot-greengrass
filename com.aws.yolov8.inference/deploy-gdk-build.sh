#!/bin/bash
set -e

AWS_ACCOUNT_NUM=$AWS_ACCOUNT_NUM
AWS_REGION=$AWS_REGION
DEV_IOT_THING=$DEV_IOT_THING
DEV_IOT_THING_GROUP=$DEV_IOT_THING_GROUP

echo "Using AWS Account $AWS_ACCOUNT_NUM in Region $AWS_REGION ..."
echo "Deploying for IoT Thing $DEV_IOT_THING in IoT Group $DEV_IOT_THING_GROUP ..."

# Setup IoT device
THING_ARN="arn:aws:iot:${AWS_REGION}:${AWS_ACCOUNT_NUM}:thing/${DEV_IOT_THING}"

# Automate the revision updates for GDK Build and Publish
pushd greengrass
if [ ! -f revision ]
then
    echo 1 > revision
fi
REVISION_VER=$(cat revision)
NEXT_REV=$((REVISION_VER+1))
echo $NEXT_REV > revision
echo Revision Version: $REVISION_VER
popd

# Set up greengrass component version
export VERSION=$(cat version)
COMPLETE_VER="$VERSION.$REVISION_VER"
VER=${COMPLETE_VER}

echo Building Component Version: $VER

jq -r --arg VER "$VER" '.component[].version=$VER' greengrass/gdk-config.json > greengrass/gdk-config.json.bak
mv greengrass/gdk-config.json.bak greengrass/gdk-config.json
jq -r --arg AWS_REGION "$AWS_REGION" '.component[].publish.region=$AWS_REGION' greengrass/gdk-config.json > greengrass/gdk-config.json.bak
mv greengrass/gdk-config.json.bak greengrass/gdk-config.json

COMPONENTCONFIGURATION=$(jq -r '.ComponentConfiguration.DefaultConfiguration' greengrass/recipe.json)

# GDK Build and GDK Publish
pushd greengrass
echo Building GDK component
gdk component build
echo Publishing GDK component
gdk component publish
popd

# GDK Component Deployment
jq -r --arg VER "$VER" '.components[].componentVersion=$VER' greengrass/deployment-config.json > greengrass/deployment-config.json.bak
mv greengrass/deployment-config.json.bak greengrass/deployment-config.json
jq -r --arg THING_ARN "$THING_ARN" '.targetArn=$THING_ARN' greengrass/deployment-config.json > greengrass/deployment-config.json.bak
mv greengrass/deployment-config.json.bak greengrass/deployment-config.json
jq -r --arg DEV_IOT_THING_GROUP "$DEV_IOT_THING_GROUP" '.deploymentName=$DEV_IOT_THING_GROUP' greengrass/deployment-config.json > greengrass/deployment-config.json.bak
mv greengrass/deployment-config.json.bak greengrass/deployment-config.json
jq -r --arg COMPONENTCONFIGURATION "$COMPONENTCONFIGURATION" '.components[].configurationUpdate.merge=$COMPONENTCONFIGURATION' greengrass/deployment-config.json > greengrass/deployment-config.json.bak
mv greengrass/deployment-config.json.bak greengrass/deployment-config.json

pushd greengrass
CONFIG_FILE="deployment-config.json"
RES=`aws greengrassv2 create-deployment --target-arn $THING_ARN --cli-input-json fileb://$CONFIG_FILE --region $AWS_REGION`
echo Greengrass Deployment ID: ${RES}
popd