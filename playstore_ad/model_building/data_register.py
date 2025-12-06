from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError
from huggingface_hub import HfApi,create_repo
import os

repo_id='Harsha1001/play-store-ad-revenue-prediction'
repo_type='dataset'

api=HfApi(token=os.getenv('HF_TOKEN_1'))

try:
  api.repo_info(repo_id=repo_id,repo_type=repo_type)
  print(f"Space '{repo_id}' already exists,using it")
except RepositoryNotFoundError:
  prin(f"Space '{repo_id}' doesn't exists,creating it")
  create_repo(repo_id=repo_id,repo_type=repo_type)
  print(f"Space '{repo_id}' created successfully")

api.upload_folder(
    folder_path='playstore_ad/data',
    repo_id=repo_id,
    repo_type=repo_type
)

