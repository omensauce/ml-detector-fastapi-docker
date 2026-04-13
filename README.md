## The project aims to show ML model deployment via a container. 

## Some notes
 - The model used is simple object detection model, that consists of pre-trained FasterRCNN with changed and tuned heads to be confroming to 3 base classes -- apples, bananas and oranges. The model training data and time was quite small, so accuracy is varying.
 - Exported format used exported weights, meaning that during inference Python needs to know architecture used, hence the existence of `model.py` file
 - The output results is located `detections` JSON array -- with fields `box` defining the bounding box in `xmin, xmax, ymin, ymax` format, `label` string defining in plaintext the non-background class, `score` probability score produced by the model
 - The model uses pre-configured NMS, the final threshold is defined in FastAPI code itself ($0.6$ is used currently)
 - Additional field is base64 encoded string representing the input image with bounding boxes marked, 

 ## Instructions

Through container (Docker/Podman)

```console
foo@bar:~$ podman build -t ml-fastapi-detector .
foo@bar:~$ podman run -p 8000:8000 ml-fastapi-detector:latest
```

Without container

```console
foo@bar:~$ uv venv ./venv
foo@bar:~$ uv pip install -r requirements.txt
foo@bar:~$ uvicorn main:app --reload
```