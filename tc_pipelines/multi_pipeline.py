
import os
import kfp
from kfp.v2.dsl import pipeline
import json

TF_TRAINER_IMAGE = os.getenv('TF_TRAINER_IMAGE')
SCIKIT_TRAINER_IMAGE = os.getenv('SCIKIT_TRAINER_IMAGE')
TORCH_TRAINER_IMAGE = os.getenv('TORCH_TRAINER_IMAGE')
XGB_TRAINER_IMAGE = os.getenv('XGB_TRAINER_IMAGE')

TF_SERVING_IMAGE = os.getenv('TF_SERVING_IMAGE')
SCIKIT_SERVING_IMAGE = os.getenv('SCIKIT_SERVING_IMAGE')
TORCH_SERVING_IMAGE = os.getenv('TORCH_SERVING_IMAGE')
XGB_SERVING_IMAGE = os.getenv('XGB_SERVING_IMAGE')

PIPELINE_ROOT = os.getenv('PIPELINE_ROOT')
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')

def get_query(dataset='training'):
    """Function that returns either training or validation query"""
    if dataset=='training':
        #split = "MOD(ABS(FARM_FINGERPRINT(CAST(customerID AS STRING))), 100) < 80"
        split = "SPLIT_TVT = 'TRAIN'"
    else:
        #split = """MOD(ABS(FARM_FINGERPRINT(CAST(customerID AS STRING))), 100) >= 80 
        #AND MOD(ABS(FARM_FINGERPRINT(CAST(customerID AS STRING))), 100) < 90"""
        split = "SPLIT_TVT = 'VALIDATION'"
        

    query = """SELECT  gender,
                       SeniorCitizen,
                       Partner,
                       Dependents,
                       tenure,
                       PhoneService,
                       MultipleLines,
                       InternetService,
                       OnlineSecurity,
                       OnlineBackup,
                       DeviceProtection,
                       TechSupport,
                       StreamingTV,
                       StreamingMovies,
                       Contract,
                       PaperlessBilling,
                       PaymentMethod,
                       MonthlyCharges,
                       COALESCE(TotalCharges,0) AS TotalCharges,
                       Churn
            FROM iorek_byrnison.telco_churn_1m_new_TVT
    WHERE {0}""".format(split)
                #FROM iorek_byrnison.telco_churn_1m
    
    return query

TRAIN_QUERY = get_query(dataset='training')
VALIDATION_QUERY=get_query(dataset='validation')

@pipeline(name='tc-kfp-multiple-pipeline',
         description='Telco Churn Multiple Framework', 
         pipeline_root=PIPELINE_ROOT)
def pipeline(
    train_query:str = TRAIN_QUERY,
    validation_query:str = VALIDATION_QUERY,
    project_id:str = PROJECT_ID ,
    region:str = REGION,
    pipeline_root:str = PIPELINE_ROOT
):
    from components.bq.extract_data import extract_data
    from components.bq.split_data import split_data
    from components.ml.sklearn_trainer import train_and_deploy as sk_train_and_deploy
    from components.ml.tensorflow_trainer import train_and_deploy as tf_train_and_deploy
    from components.ml.pytorch_trainer import train_and_deploy as torch_train_and_deploy
    from components.ml.xgboost_trainer import train_and_deploy as xgb_train_and_deploy
    
    STAGING_BUCKET = f'{PIPELINE_ROOT}/staging'
    kwargs = dict(
        project_id=project_id,
        query_job_config=json.dumps(dict(write_disposition="WRITE_TRUNCATE")),
    )
    train_split = split_data(
        query=train_query,
        dataset_id='iorek_byrnison',
        table_id='telco_churn_train_multi', 
        dataset_location=region,
        **kwargs,
    ).set_display_name('BQ Train Split')
    
    valid_split = split_data(
        query=validation_query,
        dataset_id='iorek_byrnison',
        table_id='telco_churn_valid_multi', 
        dataset_location=region,
        **kwargs,
    ).set_display_name('BQ Validation Split')
    
    train_file = (
        extract_data(
            project_id,
            'iorek_byrnison',
            'telco_churn_train_multi',
            dataset_location=region,
       )
       .after(train_split)
       .set_display_name("Extract BQ Train table to GCS")
    )
    
    valid_file = (
        extract_data(
            project_id,
            'iorek_byrnison',
            'telco_churn_valid_multi',
            dataset_location=region,
        )
        .after(valid_split)
        .set_display_name("Extract BQ Validation table to GCS")
    )
    
    tf_trainer = (
        tf_train_and_deploy(project=project_id, location = region, 
                             container_uri = TF_TRAINER_IMAGE,
                             serving_container_uri = TF_SERVING_IMAGE,
                             training_dataset = train_file.outputs["dataset"],
                             validation_dataset = valid_file.outputs["dataset"],
                             staging_bucket = STAGING_BUCKET)
            .set_display_name("Tensorflow Train and Deploy")
    )
  
    
    sk_trainer = (
        sk_train_and_deploy(project=project_id, location = region, 
                             container_uri = SCIKIT_TRAINER_IMAGE,
                             serving_container_uri = SCIKIT_SERVING_IMAGE,
                             training_dataset = train_file.outputs["dataset"],
                             validation_dataset = valid_file.outputs["dataset"],
                             staging_bucket = STAGING_BUCKET)
            .set_display_name("Sklearn Train and Deploy")
    )
    
    torch_trainer = (
        torch_train_and_deploy(project=project_id, location = region, 
                             container_uri = TORCH_TRAINER_IMAGE,
                             serving_container_uri = TORCH_SERVING_IMAGE,
                             training_dataset = train_file.outputs["dataset"],
                             validation_dataset = valid_file.outputs["dataset"],
                             staging_bucket = STAGING_BUCKET)
            .set_display_name("PyTorch Train and Deploy")
    )
    
    xgb_trainer = (
        xgb_train_and_deploy(project=project_id, location = region, 
                             container_uri = XGB_TRAINER_IMAGE,
                             serving_container_uri = XGB_SERVING_IMAGE,
                             training_dataset = train_file.outputs["dataset"],
                             validation_dataset = valid_file.outputs["dataset"],
                             staging_bucket = STAGING_BUCKET)
            .set_display_name("XGboost Train and Deploy")
    )
    
