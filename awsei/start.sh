# start the server and register the model
multi-model-server --start --mms-config config.properties --model-store modelstore
sleep 5s

curl -X POST "http://0.0.0.0:8081/models?url=resnet18.mar&batch_size=$BATCH_SIZE&max_batch_delay=$LINGER_MS"

sleep 5s
curl -X PUT "http://0.0.0.0:8081/models/resnet18?number_gpu=$GPUS&min_worker=$WORKERS"

