FROM python:3.6.10

#WORKDIR /app

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN mkdir /app 
# WORKDIR /app

RUN mkdir /app/model
RUN mkdir /app/data
RUN mkdir /app/prediction

ADD requirements.txt /
ADD data.py /
ADD model.py /
ADD get_file_list.py /

ADD pca_model.sav /app/model
ADD clf_model.sav /app/model

COPY training_data /training_data

RUN git clone https://github.com/xtracthub/xtracthub-service.git \
    && cp xtracthub-service/exceptions.py /

ENV container_version=0

RUN pip install --trusted-host pypi.python.org -r /requirements.txt

ADD test.py /
ADD xtract_images_main.py /
ENV CONTAINER_VERSION=1.0


RUN pip uninstall globus_sdk -y && pip install globus_sdk==2.0.3

RUN apt-get update && apt-get install -y libgl1-mesa-dev
WORKDIR / 
