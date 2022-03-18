
from typing import NamedTuple
from kfp.v2.dsl import Dataset, Output, component

@component(base_image='python:3.7', packages_to_install=['google-cloud-bigquery==2.30.0'])
def extract_data(
    project_id: str,
    dataset_id: str,
    table_name: str,
    dataset: Output[Dataset],
    dataset_location: str = "us-central1",
    extract_job_config: dict = None,
) -> NamedTuple("Outputs", [("dataset_gcs_prefix", str), ("dataset_gcs_uri", list)]):
    import os
    from google.cloud.exceptions import GoogleCloudError
    from google.cloud import bigquery
    from typing import NamedTuple

    full_table_id = f"{project_id}.{dataset_id}.{table_name}"

    if extract_job_config is None:
        extract_job_config = {}

    table = bigquery.table.Table(table_ref=full_table_id)
    job_config = bigquery.job.ExtractJobConfig(**extract_job_config)
    client = bigquery.client.Client(project=project_id, location=dataset_location)
    
    dataset_gcs_uri = dataset.uri
    dataset_directory = os.path.dirname(dataset_gcs_uri)
    
    extract_job = client.extract_table(
        table,
        dataset_gcs_uri,
        job_config=job_config,
    )

    try:
        result = extract_job.result()
    except GoogleCloudError as e:
        raise e

    outputs = NamedTuple("Outputs", [("dataset_gcs_prefix", str), ("dataset_gcs_uri", list)])
    ret_val = outputs(dataset_gcs_prefix=dataset_directory, dataset_gcs_uri=[dataset_gcs_uri])
    return ret_val
