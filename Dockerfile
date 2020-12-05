FROM python:3.8.6-buster

WORKDIR /root

RUN apt-get update \
    && apt-get install -y \
        git \
        gcc \
        gfortran \
        wget \
        libopenblas-base \
        libopenblas-dev \
        liblapack-dev \
        fonts-ipafont-gothic

RUN sh -c 'echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN apt-get update && apt-get install -y google-chrome-stable

COPY ./requirements.txt .
RUN pip install pip -U \
    && pip install -r requirements.txt
