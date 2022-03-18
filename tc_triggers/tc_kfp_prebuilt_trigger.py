from google.cloud import aiplatform

if __name__ == '__main__':
    
    aiplatform.init(project='qwiklabs-gcp-01-868d72e03fcc', location='us-central1')

    pipeline = aiplatform.PipelineJob(
        display_name="tc_kfp_prebuilt",
        template_path='tc_triggers/tc_kfp_prebuilt_pipeline.json',
        enable_caching=False,
    )
    pipeline.run()
