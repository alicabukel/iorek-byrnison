
from kfp.v2.dsl import Dataset, Output, component

@component(base_image='python:3.7', packages_to_install=['google-cloud-bigquery==2.30.0'])
def split_data(
    query: str,
    project_id: str,
    dataset_id: str = None,
    table_id: str = None,
    dataset_location: str = "us-central1",
    query_job_config: dict = None,
) -> None:
    from google.cloud.exceptions import GoogleCloudError
    from google.cloud import bigquery

    if (dataset_id is not None) and (table_id is not None):
        dest_table_ref = f"{project_id}.{dataset_id}.{table_id}"
    else:
        dest_table_ref = None
    if query_job_config is None:
        query_job_config = {}
    bq_client = bigquery.client.Client(project=project_id, location=dataset_location)
    job_config = bigquery.QueryJobConfig(destination=dest_table_ref, **query_job_config)
    query_job = bq_client.query(query, job_config=job_config)

    try:
        result = query_job.result()
    except GoogleCloudError as e:
        raise e
