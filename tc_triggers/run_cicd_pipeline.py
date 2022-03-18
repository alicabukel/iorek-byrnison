import fire
from google.cloud import aiplatform


def run_pipeline(project_id, region, template_path, display_name):

    aiplatform.init(project=project_id, location=region)

    pipeline = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=template_path,
        enable_caching=False,
    )

    pipeline.run()


if __name__ == "__main__":
    fire.Fire(run_pipeline)