import os
from datetime import datetime

from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.aiplatform import (
    EndpointCreateOp,
    ModelDeployOp,
    ModelUploadOp,
)
from google_cloud_pipeline_components.experimental import (
    hyperparameter_tuning_job,
)
from google_cloud_pipeline_components.experimental.custom_job import (
    CustomTrainingJobOp,
)
from kfp.v2 import dsl

PIPELINE_ROOT = os.getenv('PIPELINE_ROOT', 'gs://iorek-byrnison/tc-cicd-dir/artfact-store/kfp-prebuilt')
PROJECT_ID = os.getenv('PROJECT_ID', 'qwiklabs-gcp-01-868d72e03fcc')
REGION = os.getenv('REGION', 'us-central1')

TRAINING_CONTAINER_IMAGE_URI = os.getenv('TRAINING_CONTAINER_IMAGE_URI', 
                                         'gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_kfp_sklearn_trainer:latest')

SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI",
                                       "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-20:latest")

SERVING_MACHINE_TYPE = os.getenv("SERVING_MACHINE_TYPE", "n1-standard-16")

TRAINING_FILE_PATH = os.getenv('TRAINING_FILE_PATH',
                               'gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train1m.csv')
VALIDATION_FILE_PATH = os.getenv('VALIDATION_FILE_PATH',
                                 'gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid1m.csv')

MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "5"))
PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "5"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", f'{PIPELINE_ROOT}/{TIMESTAMP}')


@dsl.pipeline(
    name="tc-kfp-prebuilt-pipeline",
    description="Kubeflow pipeline that tunes, trains, and deploys on Vertex",
    pipeline_root=PIPELINE_ROOT,
)
def create_pipeline():

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINING_CONTAINER_IMAGE_URI,
                "args": [
                    f"--training_dataset_path={TRAINING_FILE_PATH}",
                    f"--validation_dataset_path={VALIDATION_FILE_PATH}",
                    "--hptune",
                ],
            },
        }
    ]

    metric_spec = hyperparameter_tuning_job.serialize_metrics(
        {"accuracy": "maximize"}
    )

    parameter_spec = hyperparameter_tuning_job.serialize_parameters(
        {
            "alpha": hpt.DoubleParameterSpec(
                min=1.0e-4, max=1.0e-1, scale="linear"
            ),
            "max_iter": hpt.DiscreteParameterSpec(
                values=[1, 2], scale="linear"
            ),
        }
    )

    hp_tuning_task = hyperparameter_tuning_job.HyperparameterTuningJobRunOp(
        display_name="tc-kfp-tuning-prebuilt-job",
        project=PROJECT_ID,
        location=REGION,
        worker_pool_specs=worker_pool_specs,
        study_spec_metrics=metric_spec,
        study_spec_parameters=parameter_spec,
        max_trial_count=MAX_TRIAL_COUNT,
        parallel_trial_count=PARALLEL_TRIAL_COUNT,
        base_output_directory=PIPELINE_ROOT,
    )

    trials_task = hyperparameter_tuning_job.GetTrialsOp(
        gcp_resources=hp_tuning_task.outputs["gcp_resources"], #region=REGION
    )

    best_hyperparameters_task = (
        hyperparameter_tuning_job.GetBestHyperparametersOp(
            trials=trials_task.output, study_spec_metrics=metric_spec
        )
    )

    worker_pool_specs_task = hyperparameter_tuning_job.GetWorkerPoolSpecsOp(
        best_hyperparameters=best_hyperparameters_task.output,
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": TRAINING_CONTAINER_IMAGE_URI,
                    "args": [
                        f"--training_dataset_path={TRAINING_FILE_PATH}",
                        f"--validation_dataset_path={VALIDATION_FILE_PATH}",
                        "--nohptune",
                    ],
                },
            }
        ],
    )

    training_task = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name="tc-kfp-training-prebuilt-job",
        worker_pool_specs=worker_pool_specs_task.output,
        base_output_directory=BASE_OUTPUT_DIR,
    )

    model_upload_task = ModelUploadOp(
        project=PROJECT_ID,
        display_name=f"tc-kfp-prebuilt-model-upload-job",
        artifact_uri=f"{BASE_OUTPUT_DIR}/model",
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
    )
    model_upload_task.after(training_task)

    endpoint_create_task = EndpointCreateOp(
        project=PROJECT_ID,
        display_name="tc-kfp-prebuilt-create-endpoint-job",
    )
    endpoint_create_task.after(model_upload_task)

    model_deploy_op = ModelDeployOp(
        model=model_upload_task.outputs["model"],
        endpoint=endpoint_create_task.outputs["endpoint"],
        deployed_model_display_name='tc-kfp-prebuilt-model',
        dedicated_resources_machine_type=SERVING_MACHINE_TYPE,
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )
