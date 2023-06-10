#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure that the model path is set
if [ -z $CONTAINER_MODEL_PATH ]; then
  echo "Must specify model path"
  exit 1
fi

# Ensure that bind-mounted directories are owned by the user that runs the service
chown -R $RUN_UID:$RUN_UID $CONTAINER_MODEL_PATH
chown -R $RUN_UID:$RUN_UID /home/user/exllama_sessions

# Run service as specified (non-root) user
exec runuser -u $(id -un $RUN_UID) -- python3 /app/webui/app.py -d $CONTAINER_MODEL_PATH $@
