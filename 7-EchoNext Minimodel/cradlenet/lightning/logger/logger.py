from azureml.core.run import Run, _OfflineRun
from pytorch_lightning.loggers import MLFlowLogger


def get_logger():
    run = Run.get_context()
    if isinstance(run, _OfflineRun):
        logger = MLFlowLogger()
    else:
        mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
        logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
        logger._run_id = run.id
    return logger
