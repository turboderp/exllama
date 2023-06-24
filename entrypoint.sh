#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure that the application state path is set
if [ -z $APPLICATION_STATE_PATH ]; then
  echo "Must specify application state path"
  exit 1
fi

# Ensure that bind-mounted directories are owned by the user that runs the service if the user is not root
if [ $RUN_UID -ne 0 ]; then
  chown -R $RUN_UID:$RUN_UID $APPLICATION_STATE_PATH
fi

# Run service as specified (non-root) user
exec runuser -u $(id -un $RUN_UID) -- python3 /app/webui/app.py \
	-d $CONTAINER_MODEL_PATH \
	--sessions_dir $CONTAINER_SESSIONS_PATH \
	$@
