
if __name__ == '__main__':
    from google.cloud import aiplatform
    aiplatform.init(project='qwiklabs-gcp-01-868d72e03fcc', location='us-central1')

    pipeline = aiplatform.PipelineJob(
        display_name="tc_kfp_lightweight",
        template_path='tc_triggers/tc_kfp_lightweight_pipeline.json',
        enable_caching=False,
    )
    pipeline.run()