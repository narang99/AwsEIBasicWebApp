FROM awsdeeplearningteam/multi-model-server

USER root

RUN ["/bin/bash", "-c", "python3 -m pip install --upgrade pip"]
RUN ["/bin/bash", "-c", "pip3 install torch==1.5.1 torchvision==0.6.1 --extra-index-url https://download.pytorch.org/whl/cpu"]

ENV BATCH_SIZE=32
ENV LINGER_MS=5
ENV WORKERS=1
ENV GPUS=0

WORKDIR /home/model-server/custom

# create the archive
COPY resnet resnet
COPY config.properties .
COPY start.sh .
COPY modelstore modelstore

RUN ["/bin/bash", "-c", "model-archiver --handler batch:handle --model-path resnet/ --export-path modelstore --model-name resnet18"]
RUN chmod 700 start.sh

EXPOSE 8080
EXPOSE 8081

# start the server and register the model
CMD ./start.sh $BATCH_SIZE $LINGER_MS $WORKERS $GPUS

