from google.cloud import aiplatform

if __name__ == '__main__':
    aiplatform.init(project='qwiklabs-gcp-01-868d72e03fcc', location='us-central1')

    pipeline = aiplatform.PipelineJob(
        display_name='tc-tfx-pipeline',
        template_path="tc-tfx/pipeline.json",
        #enable_caching=True,
    )

    pipeline.run()