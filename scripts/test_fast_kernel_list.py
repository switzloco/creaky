from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as client:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = "nswitzer"
    req.kernel_slug = "creaky-caching-dataset-n4-384"
    req.page_size = 100
    resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
    print(f"Got {len(resp.files)} files on page 1, next_page_token={bool(resp.next_page_token)}")
    if resp.files:
        print("Sample file_name:", resp.files[0].file_name)
