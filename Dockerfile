FROM python:3.6.3

WORKDIR /app


RUN mkdir model
RUN mkdir data
RUN mkdir prediction

ADD requirements.txt /app
ADD xtract_images_main.py /app
ADD data.py /app
ADD model.py /app
ADD get_file_list.py /app

ADD pca_model.sav /app
ADD clf_model.sav /app

COPY training_data /app/training_data

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install git+https://github.com/Parsl/parsl
RUN pip3 install git+https://github.com/DLHub-Argonne/home_run