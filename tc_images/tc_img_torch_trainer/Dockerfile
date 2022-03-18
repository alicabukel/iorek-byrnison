
FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire torch==1.6.0 scikit-learn==0.23.2 pandas==1.1.1
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
