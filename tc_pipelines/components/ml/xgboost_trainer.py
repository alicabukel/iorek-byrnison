from kfp.v2.dsl import component, Input, Dataset
@component(
    base_image="python:3.7",
    packages_to_install=["google-cloud-aiplatform==1.7.1"],
)
def train_and_deploy(
    project: str,
    location: str,
    container_uri: str,
    serving_container_uri: str,
    training_dataset: Input[Dataset],
    validation_dataset: Input[Dataset],
    staging_bucket: str,
):

    from google.cloud import aiplatform
    
    aiplatform.init(
        project=project, location=location, staging_bucket=staging_bucket
    )
    job = aiplatform.CustomContainerTrainingJob(
        display_name="telco_churn_kfp_training_xgb",
        container_uri=container_uri,
        command=[
            "python",
            "train.py",
            f"--training_dataset_path={training_dataset.uri}",
            f"--validation_dataset_path={validation_dataset.uri}",
            '--max_depth', '10', 
            '--n_estimators', '100',
        ],
        staging_bucket=staging_bucket,
        model_serving_container_image_uri=serving_container_uri,
    )
    model = job.run(replica_count=1, model_display_name="telco_churn_kfp_model_xgb")
    endpoint = model.deploy( 
        traffic_split={"0": 100},
        machine_type="n1-standard-2",
    )
