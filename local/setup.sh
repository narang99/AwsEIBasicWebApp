#!/bin/bash

SERVER_URL=$1
if [ -z "$SERVER_URL" ]; then
  echo "Please provide IP:port or domain of the model server as first argument. Exiting"
  exit 1
fi

cat ./template/index.html | sed "s/{{SERVER_LINK}}/$SERVER_URL/" > index.html
