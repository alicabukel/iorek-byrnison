
FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire scikit-learn==0.23.2 pandas==1.1.1 xgboost==1.2.0
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
