FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-eia:1.5.1-cpu-py36-ubuntu16.04

ENV BATCH_SIZE=32
ENV LINGER_MS=5
ENV WORKERS=1
ENV GPUS=1

WORKDIR /qure

# create the archive
COPY modelstore modelstore
COPY resnet resnet
COPY config.properties .
COPY start.sh .

RUN model-archiver --handler batch:handle --model-path resnet/ --export-path modelstore/ --model-name resnet18
RUN chmod 700 start.sh

EXPOSE 8080
EXPOSE 8081

# start the server and register the model
CMD ./start.sh $BATCH_SIZE $LINGER_MS $WORKERS $GPUS
