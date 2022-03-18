def train_and_deploy(
    project: str,
    location: str,
    container_uri: str,
    serving_container_uri: str,
    training_file_path: str,
    validation_file_path: str,
    staging_bucket: str,
    alpha: float,
    max_iter: int,
):

    from google.cloud import aiplatform

    aiplatform.init(
        project=project, location=location, staging_bucket=staging_bucket
    )
    job = aiplatform.CustomContainerTrainingJob(
        display_name="churn_kfp_training",
        container_uri=container_uri,
        command=[
            "python",
            "train.py",
            f"--training_dataset_path={training_file_path}",
            f"--validation_dataset_path={validation_file_path}",
            f"--alpha={alpha}",
            f"--max_iter={max_iter}",
            "--nohptune",
        ],
        staging_bucket=staging_bucket,
        model_serving_container_image_uri=serving_container_uri,
    )
    model = job.run(replica_count=1, model_display_name="churn_kfp_model")
    endpoint = model.deploy( 
        traffic_split={"0": 100},
        machine_type="n1-standard-2",
    )
