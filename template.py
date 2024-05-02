import logging
import os
from pathlib import Path
logging.basicConfig(level=logging.INFO)
project_name = "mlops_project"
file_list = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_tranformation.py",
    f"src/{project_name}/components/data_training.py",
    f"src/{project_name}/components/data_monotering.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/train_pipeline.py",
    f"src/{project_name}/pipelines/test_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "app.py"

]

for file_path in file_list:
    file_path = Path(file_path)
    file_dirs,file_names = os.path.split(file_path)
    if file_dirs!="":
        os.makedirs(file_dirs, exist_ok=True)
        logging.info(f"Creating {file_dirs} for {file_names}")
    if (not os.path.exists(file_path) or (os.path.getsize(file_path)==0)):
        with open(file_path,'w') as file:
            pass
    else:
        logging.info(f"File {file_path} already exists")