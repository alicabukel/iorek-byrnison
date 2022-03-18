import os

from kfp import dsl
from kfp.components import create_component_from_func_v2
from components.training import train_and_deploy
from components.tuning import tune_hyperparameters


TRAINING_CONTAINER_IMAGE_URI = os.getenv('TRAINING_CONTAINER_IMAGE_URI', 
                                         'gcr.io/qwiklabs-gcp-01-868d72e03fcc/tc_img_kfp_sklearn_trainer:latest')

SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI",
                                       "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-20:latest")

TRAINING_FILE_PATH = os.getenv('TRAINING_FILE_PATH',
                               'gs://iorek-byrnison/tc-cicd-dir/data/tc-data/train1m.csv')
VALIDATION_FILE_PATH = os.getenv('VALIDATION_FILE_PATH',
                                 'gs://iorek-byrnison/tc-cicd-dir/data/tc-data/valid1m.csv')

MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "5"))
PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "5"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))


PIPELINE_ROOT = os.getenv('PIPELINE_ROOT', 'gs://iorek-byrnison/tc-cicd-dir/artfact-store/kfp-lightweight')
PROJECT_ID = os.getenv('PROJECT_ID', 'qwiklabs-gcp-01-868d72e03fcc')
REGION = os.getenv('REGION', 'us-central1')


tune_hyperparameters_component = create_component_from_func_v2(
    tune_hyperparameters,
    base_image="python:3.8",
    output_component_file="tc_kfp_lightweight_tuner.yaml",
    packages_to_install=["google-cloud-aiplatform"],
)


train_and_deploy_component = create_component_from_func_v2(
    train_and_deploy,
    base_image="python:3.8",
    output_component_file="tc_kfp_lightweight_trainer.yaml",
    packages_to_install=["google-cloud-aiplatform"],
)

@dsl.pipeline(
    name="tc-kfp-lightweight-pipeline",
    description="The pipeline training and deploying the Telco Churn classifier",
    pipeline_root=PIPELINE_ROOT,
)
def churn_train(
    training_container_uri: str = TRAINING_CONTAINER_IMAGE_URI,
    serving_container_uri: str = SERVING_CONTAINER_IMAGE_URI,
    training_file_path: str = TRAINING_FILE_PATH,
    validation_file_path: str = VALIDATION_FILE_PATH,
    accuracy_deployment_threshold: float = THRESHOLD,
    max_trial_count: int = MAX_TRIAL_COUNT,
    parallel_trial_count: int = PARALLEL_TRIAL_COUNT,
    pipeline_root: str = PIPELINE_ROOT,
):
    staging_bucket = f"{pipeline_root}/staging"

    tuning_op = tune_hyperparameters_component(
        project=PROJECT_ID,
        location=REGION,
        container_uri=training_container_uri,
        training_file_path=training_file_path,
        validation_file_path=validation_file_path,
        staging_bucket=staging_bucket,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
    )

    accuracy = tuning_op.outputs["best_accuracy"]

    with dsl.Condition(
        accuracy >= accuracy_deployment_threshold, name="deploy_decision"
    ):
        train_and_deploy_op = ( 
            train_and_deploy_component(
                project=PROJECT_ID,
                location=REGION,
                container_uri=training_container_uri,
                serving_container_uri=serving_container_uri,
                training_file_path=training_file_path,
                validation_file_path=validation_file_path,
                staging_bucket=staging_bucket,
                alpha=tuning_op.outputs["best_alpha"],
                max_iter=tuning_op.outputs["best_max_iter"],
            )
        )