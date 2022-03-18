# IorekByrnison

1. [Vertex Training Jobs](#vertex-training-jobs)
    - a) [Module Design](#mdule-design)
    - b) [Local Tests](#local-tests)
    - c) [Packaging for Deployment](#packaging-for-deployment)
    - d) [Deployment with Prebuilt Image](#deployment-with-prebuilt-image)
    - e) [Building Custom Image](#building-custom-image)
    - f) [Deployment with Custom Image](#deployment-with-custom-image)
    - g) [Hyperparameter Tuning Job](#hyperparameter-tuning-job)
2. [Kubeflow Pipelines](#kubeflow-pipelines)    
    - a) [Module Architecture](#module-architecture)
    - b) [Building Trainer Image](#building-trainer-image)
    - c) [Lightweight Components](#lightweight-components)
    - d) [Prebuilt Components](#prebuilt-components)
    - e) [Building Trigger Image](#building-trigger-image)
    - f) [Cloud Build Triggers](#cloud-build-triggers)
3. [Multiple Training with KFP](#multiple-training-with-kfp)
    - a) [Module Overview](#module-overview)
    - b) [Building Trainer Images](#building-trainer-images)
    - c) [Executing Pipeline](#executing-pipeline)
    - d) [Github Triggers](#github-triggers)
4. [TFX Pipelines](#tfx-pipelines)
    - a) [Scaffolding](#scaffolding)
    - b) [Unit Tests](#unit-tests)
    - c) [Building TFX Image](#building-tfx-image)
    - d) [Deploying TFX Pipeline](#deploying-tfx-pipeline)
    


### Vertex Training Jobs

##### References

[Keras Sequential API](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/introduction_to_tensorflow/solutions/3_keras_sequential_api_vertex.ipynb) | [Keras Functional API](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/introduction_to_tensorflow/solutions/4_keras_functional_api.ipynb) | [Keras Feature Engineering](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/feature_engineering/solutions/4_keras_adv_feat_eng.ipynb) | [Training at Scale](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/building_production_ml_systems/solutions/1_training_at_scale_vertex.ipynb) | [Hyperparameter Tuning](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/building_production_ml_systems/solutions/2_hyperparameter_tuning_vertex.ipynb) | [Keras for Text Classification](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/text_models/solutions/keras_for_text_classification.ipynb) | [Keras Time Series Modelling](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/time_series_prediction/solutions/4_modeling_keras.ipynb)

##### Resources

- Training Data: gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train.csv
- Validation Data: gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid.csv
- Training Data: gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train1m.csv
- Validation Data: gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid1m.csv
- Python Package: gs://iorek-byrnison/tc-cicd-dir/bin/tc_trainer-0.1.tar.gz
- Training Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_custom_ai_trainer:latest

#### Module Design

```
- data
- tc_trainer/
    + Dockerfile
    + setup.py
    + config.yaml
    + trainer/
        + constants.py
        + data.py
        + transform.py
        + model.py
        + train.py
        + tuner.py
        + ml_task.py
        + hpt_task.py
        + __init__.py
```

#### Local Tests

```shell

OUTPUT_DIR=./tc_tainer_out 
test ${OUTPUT_DIR} && rm -rf ${OUTPUT_DIR}
export PYTHONPATH=${PYTHONPATH}:${PWD}/tc_trainer
    
python3 -m trainer.ml_task \
--train_data_path ./data/train.csv \
--eval_data_path ./data/valid.csv \
--output_dir $OUTPUT_DIR \
--batch_size 5 \
--num_examples_to_train_on 100 \
--num_evals 1 \
--nbuckets 10 \
--lr 0.001 \
--nnsize "32 8"

```

#### Packaging for Deployment

```shell

cd tc_trainer/

python ./setup.py sdist --formats=gztar

cd ..

gsutil cp tc_trainer/dist/tc_trainer-0.1.tar.gz gs://iorek-byrnison/tc-cicd-dir/bin/

```

#### Deployment with Prebuilt Image

```shell

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

WORKER_POOL_SPEC="machine-type=n1-standard-4,\
replica-count=1,\
executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-5:latest,\
python-module=trainer.ml_task"


ARGS="--train_data_path=gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train.csv,\
--eval_data_path=gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid.csv,\
--output_dir=gs://iorek-byrnison/tc-cicd-dir/out/trained_model_$TIMESTAMP,\
--batch_size=50,\
--num_examples_to_train_on=5000,\
--num_evals=100,\
--nbuckets=10,\
--lr=0.001,\
--nnsize=32 8"

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=tc_custom_ai_job_$TIMESTAMP \
  --python-package-uris=gs://iorek-byrnison/tc-cicd-dir/bin/tc_trainer-0.1.tar.gz \
  --worker-pool-spec=$WORKER_POOL_SPEC \
  --args="$ARGS"
  
```

```shell

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

WORKER_POOL_SPEC="machine-type=n1-standard-4,\
replica-count=1,\
executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-5:latest,\
python-module=trainer.ml_task"


ARGS="--train_data_path=gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train1m.csv,\
--eval_data_path=gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid1m.csv,\
--output_dir=gs://iorek-byrnison/tc-cicd-dir/out/trained_model_$TIMESTAMP,\
--batch_size=50,\
--num_examples_to_train_on=5000,\
--num_evals=100,\
--nbuckets=10,\
--lr=0.001,\
--nnsize=32 8"

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=tc_custom_ai_job_$TIMESTAMP \
  --python-package-uris=gs://iorek-byrnison/tc-cicd-dir/bin/tc_trainer-0.1.tar.gz \
  --worker-pool-spec=$WORKER_POOL_SPEC \
  --args="$ARGS"
 
```

```shell

tensorboard dev upload \
    --logdir gs://iorek-byrnison/tc-cicd-dir/out/trained_model_20220317_071952 \
    --name "Telco Churn - ASL First Model" \
    --description "Simple Custom AI Training Job"

```

#### Building Custom Image

```shell

docker build tc_trainer -f tc_trainer/Dockerfile \
    -t gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_custom_ai_trainer:latest

docker push gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_custom_ai_trainer:latest

```

#### Deployment with Custom Image

```shell

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

WORKER_POOL_SPEC="machine-type=n1-standard-4,\
replica-count=1,\
container-image-uri=gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_custom_ai_trainer"

ARGS="--train_data_path=gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train1m.csv,\
--eval_data_path=gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid1m.csv,\
--output_dir=gs://iorek-byrnison/tc-cicd-dir/out/trained_model_$TIMESTAMP,\
--batch_size=50,\
--num_examples_to_train_on=5000,\
--num_evals=100,\
--nbuckets=10,\
--lr=0.001,\
--nnsize=32 8"

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=tc_custom_ai_container_job_$TIMESTAMP \
  --worker-pool-spec=$WORKER_POOL_SPEC \
  --args="$ARGS"

```

```shell

tensorboard dev upload \
    --logdir gs://iorek-byrnison/tc-cicd-dir/out/trained_model_20220317_081921 \
    --name "Telco Churn - ASL Second Model" \
    --description "Simple Custom AI Training Job with Custom Containers"

```

#### Hyperparameter Tuning Job

```shell

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

gcloud beta ai hp-tuning-jobs create \
    --region=us-central1 \
    --display-name=tc_custom_ai_hpt_job_$TIMESTAMP \
    --config=tc_trainer/config.yaml \
    --max-trial-count=10 \
    --parallel-trial-count=2
    
```

```shell

tensorboard dev upload \
    --logdir gs://iorek-byrnison/tc-cicd-dir/out/tc_churn_hpt_model1/6 \
    --name "Telco Churn - ASL HPT Model" \
    --description "Simple HPT AI Job"

```

### Kubeflow Pipelines

##### References

[Kubeflow Pipelines](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/building_production_ml_systems/solutions/3_kubeflow_pipelines_vertex.ipynb) | [Kubeflow Pipelines Walkthrough](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/kubeflow_pipelines/walkthrough/solutions/kfp_walkthrough_vertex.ipynb) | [KFP AutoML](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/kubeflow_pipelines/pipelines/solutions/kfp_pipeline_vertex_automl.ipynb) | [KFP Lightweight Components](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/kubeflow_pipelines/pipelines/solutions/kfp_pipeline_vertex.ipynb) | [KFP Pre-built Components](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/kubeflow_pipelines/pipelines/solutions/kfp_pipeline_vertex_prebuilt.ipynb) | [KFP CI/CD Pipelines](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/kubeflow_pipelines/cicd/solutions/kfp_cicd_vertex.ipynb)

##### Resources

- Training Data: gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train1m.csv
- Validation Data: gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid1m.csv
- Trainer Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_kfp_sklearn_trainer:latest
- Trigger Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_kfp_trigger_cicd:latest

#### Module Architecture

```
- tc_images/
    + tc_img_kfp_sklearn_trainer/
        + Dockerfile
        + train.py
    + tc_img_kfp_trigger_cicd/
        + Dockerfile
- tc_pipelines/
    + lightweight_pipeline.py
    + prebuilt_pipeline.py
    + automl_pipeline.py
    + __init__.py
    + components/
        + training.py
        + tuning.py
        + __init__.py
- tc_triggers/
    + tc_kfp_lightweight_trigger.py
    + tc_kfp_lightweight_pipeline.json
    + tc_kfp_prebuilt_trigger.py
    + tc_kfp_prebuilt_pipeline.json
    + lightweight.cloudbuild.yaml
    + prebuilt.cloudbuild.yaml
    + run_cicd_pipeline.py
- tc_triggers/
    + multi.cloudbuild.yaml
    + tc_kfp_multi_trigger.py
    
```

#### Building Trainer Image

```shell

gcloud builds submit --timeout 15m \
    --tag gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_kfp_sklearn_trainer:latest \
    tc_images/tc_img_kfp_sklearn_trainer

```

#### Lightweight Components

```shell

dsl-compile-v2 --py tc_pipelines/lightweight_pipeline.py \
    --output tc_triggers/tc_kfp_lightweight_pipeline.json

python tc_triggers/tc_kfp_lightweight_trigger.py

```


#### Prebuilt Components

```shell

dsl-compile-v2 --py tc_pipelines/prebuilt_pipeline.py \
    --output tc_triggers/tc_kfp_prebuilt_pipeline.json

python tc_triggers/tc_kfp_prebuilt_trigger.py
   
```

#### Building Trigger Image

```shell

gcloud builds submit --timeout 15m \
    --tag gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_kfp_trigger_cicd:latest \
    tc_images/tc_img_kfp_trigger_cicd

```

#### Cloud Build Triggers

```shell

gcloud builds submit . \
    --config tc_triggers/lightweight.cloudbuild.yaml \
    --substitutions _REGION=us-central1,_PIPELINE_FOLDER=./

```

```shell

gcloud builds submit . \
    --config tc_triggers/prebuilt.cloudbuild.yaml \
    --substitutions _REGION=us-central1,_PIPELINE_FOLDER=./

```

### Multiple Training with KFP

##### References

[KFP Multiple Frameworks](https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/master/immersion/kubeflow_pipelines/multiple_frameworks/solutions/lab-01.ipynb)

##### Resources

- Trainer Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_sk_trainer:latest
- Trainer Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_xgb_trainer:latest
- Trainer Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_tf_trainer:latest
- Trainer Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_torch_trainer:latest


#### Module Overview

```
- tc_images/
    + tc_img_sk_trainer/
        + Dockerfile
        + train.py
    + tc_img_xgb_trainer/
        + Dockerfile
        + train.py
    + tc_img_tf_trainer/
        + Dockerfile
        + train.py
    + tc_img_torch_trainer/
        + Dockerfile
        + train.py
- tc_pipelines/
    + multi_pipeline.py
    + __init__.py
    + components/
        + __init__.py
        + bq/
            + split_data.py
            + extract_data.py
            + __init__.py
        + ml/
            + sklearn_trainer.py
            + xgboost_trainer.py
            + tensorflow_trainer.py
            + pytorch_trainer.py
            + __init__.py
```

#### Building Trainer Images

```shell

gcloud builds submit \
    --tag gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_sk_trainer:latest \
    tc_images/tc_img_sk_trainer

```

```shell

gcloud builds submit \
    --tag gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_xgb_trainer:latest \
    tc_images/tc_img_xgb_trainer

```

```shell

gcloud builds submit \
    --tag gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_tf_trainer:latest \
    tc_images/tc_img_tf_trainer

```

```shell

gcloud builds submit \
    --tag gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_torch_trainer:latest \
    tc_images/tc_img_torch_trainer

```

#### Executing Pipeline

```shell

export PROJECT_ID=qwiklabs-gcp-01-868d72e03fcc
export REGION=us-central1
export BUCKET=iorek-byrnison
export PIPELINE_ROOT=gs://iorek-byrnison/tc-cicd-dir/kfp-multi

export SCIKIT_TRAINER_IMAGE=gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_sk_trainer:latest
export XGB_TRAINER_IMAGE=gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_xgb_trainer:latest
export TF_TRAINER_IMAGE=gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_tf_trainer:latest
export TORCH_TRAINER_IMAGE=gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_torch_trainer:latest

export SCIKIT_SERVING_IMAGE=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-20:latest
export XGB_SERVING_IMAGE=us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-2:latest
export TF_SERVING_IMAGE=us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-1:latest
export TORCH_SERVING_IMAGE=us-docker.pkg.dev/vertex-ai/prediction/pytorch-xla.1-6:latest

dsl-compile-v2 --py tc_pipelines/multi_pipeline.py \
    --output tc_triggers/tc_kfp_multi_pipeline.json

python tc_triggers/tc_kfp_multi_trigger.py

```

#### Github Triggers

```shell

gcloud builds submit . \
    --config tc_triggers/multi.cloudbuild.yaml \
    --substitutions _REGION=us-central1,_PIPELINE_FOLDER=./

```

Cloud Build
=====

- TRIGGER SETTINGS
    + Name: sample-trigger
    + Description: Invokes a build every time code is pushed to any branch
    + Event: Tag
    + Source: [YOUR FORK]
    + Tag (regex): .*
    + Build Configuration: Cloud Build configuration file (yaml or json)
    + Cloud Build configuration file location: tc_triggers/multi.cloudbuild.yaml
- VARIABLES
    + _REGION=us-central1
    + _PIPELINE_FOLDER=.

```shell

git tag [TAG NAME]
git push origin --tags

```


### TFX Pipelines

##### References

[TFX Walkthrough](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/tfx_pipelines/walkthrough/solutions/tfx_walkthrough_vertex.ipynb) | [TFX Pipelines](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/tfx_pipelines/pipeline/solutions/tfx_pipeline_vertex.ipynb) | [TFX CI/CD](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/tfx_pipelines/cicd/solutions/tfx_cicd_vertex.ipynb) | [TX Guided Project](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/tfx_pipelines/guided_projects/tfx_guided_project_vertex.ipynb)

##### Resources

Trainer Image: gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_tfx:latest

#### Scaffolding

```shell

tfx template copy \
  --pipeline-name=tc-tfx \
  --destination-path=./tc-tfx \
  --model=taxi #penguin

```

```

- tc_tfx/
    + kubeflow_v2_runner.py
    + kubeflow_runner.py
    + local_runner.py
    + kubeflow_v2_runner.py
    + __init__.py
    + data_validation.ipynb
    + model_analysis.ipynb
    + Dockerfile
    + pipeline.json
    + data/
        + data.csv
    + pipeline/
        + pipeline.py
        + configs.py
        + __init__.py
    + models/
        + features.py
        + features_test.py
        + preprocessing.py
        + preprocessing_test.py
        + __init__.py
        + keras_model/
            + model.py
            + model_test.py
            + constants.py
            + __init__.py
        + estimator_model/
            + model.py
            + model_test.py
            + constants.py
            + __init__.py

```


#### Unit Tests

```shell
cd tc-tfx

python -m models.features_test
python -m models.keras_model.model_test

cd ..

```

#### Building TFX Image

```shell

gcloud builds submit --timeout 15m \
    --tag gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_tfx:latest \
    tc-tfx

```

#### Deploying TFX Pipeline


```shell

cd tc-tfx

tfx pipeline compile --engine vertex --pipeline_path kubeflow_v2_runner.py

cd ..

python tc_triggers/tc_tfx_trigger.py

```
