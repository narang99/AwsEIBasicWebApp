FROM python

ENV SERVER_LINK=localhost:8080

WORKDIR /frontend
COPY template template
COPY setup.sh .
COPY placeholder.png .

RUN chmod 700 setup.sh
RUN ./setup.sh $SERVER_LINK
CMD ["python3", "-m", "http.server"]
