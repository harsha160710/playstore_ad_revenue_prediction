from huggingface_hub import HfApi 
import os

api=HfApi(token=os.getenv("hFz_TOKEN_1"))
api.upload_folder(
    folder_path='playstore_ad/deployment',
    repo_id='Harsha1001/play-store-ad-revenue-prediction/playstore_revenue_model',
    repo_type='space',
    path_in_repo=""
)
