#FROM jupyter/scipy-notebook - commented out for debugging purposes 
FROM python:3.9.16-bullseye

# Loading all required files 
COPY requirements.txt ./requirements.txt
COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY app.py ./app.py

# Installing the dependencies 
RUN pip install -r requirements.txt

# Running the training
RUN python3 train.py

# Running the test file
RUN python3 inference.py 