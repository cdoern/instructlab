
from instructlab.api.schema.app import app
from flask import request, jsonify
from instructlab.model.download import download

@app.post("/model/download")
def download():
    args = request.get_json()
    model_name = args["model_name"]
    repo = args["repo"]
    release = args["release"]
    filename = args["filename"]
    model_dir = args["model_dir"]
    hf_token = args["hf_token"]
    ctx = create_click_context()
    download(
        repository=repo,
        release=release,
        filename=filename,
        model_dir=model_dir,
        hf_token=hf_token,
    )


