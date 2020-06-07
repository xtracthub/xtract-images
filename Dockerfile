FROM python:3.6.3

WORKDIR /app


RUN mkdir model
RUN mkdir data
RUN mkdir prediction

ADD requirements.txt /app
ADD data.py /app
ADD model.py /app
ADD get_file_list.py /app

ADD pca_model.sav /app
ADD clf_model.sav /app

COPY training_data /app/training_data

RUN git clone https://github.com/xtracthub/xtracthub-service.git \
    && cp xtracthub-service/exceptions.py /


ENV container_version=0
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
ADD xtract_images_main.py /app
