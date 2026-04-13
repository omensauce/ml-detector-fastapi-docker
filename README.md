## Context
This small project aims to show ML model deployment via a container. 

## Some notes
 - The model used is simple object detection model, that consists of pre-trained FasterRCNN with changed and tuned heads to be confroming to 3 base classes &ndash; apples, bananas and oranges. The model training data was quite small, so accuracy of the model is varying
 - Exported format used exported weights, meaning that during inference Python needs to know architecture used, hence the existence of `model.py` file
 - The output results is located `detections` JSON array &ndash; with fields `box` defining the bounding box in `xmin, xmax, ymin, ymax` format, `label` string defining in plaintext the non-background class, `score` probability score produced by the model
 - The model uses pre-configured NMS, the final threshold is defined in FastAPI code itself ($0.6$ is used currently)
 - Additional field `image` is base64 encoded string representing the input image with bounding boxes marked, hacky way to see it is to type `data:image/jpeg;base64,STRING`, where `STRING` is the encoded value (without quotation marks 

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


## Example

The image of apple is used ![apple](https://github.com/user-attachments/assets/1f28963d-700c-43df-af13-f8cbf12637c2)


The request command:

```
curl -X 'POST' \
  'http://172.31.125.30:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@apple.jpg;type=image/jpeg'
```

The response looks like following (with difference that for real response image section is omitted here for clarity, the image is attached below): 

```JSON
{
  "detections": [
    {
      "box": [
        1940.8443603515625,
        137.0790252685547,
        3719.707763671875,
        1904.5316162109375
      ],
      "label": "apple",
      "score": 0.9913695454597473
    },
    {
      "box": [
        1015.9450073242188,
        820.731689453125,
        3042.80078125,
        2934.6533203125
      ],
      "label": "apple",
      "score": 0.9885911345481873
    },
    {
      "box": [
        304.6193542480469,
        224.8274688720703,
        1795.1361083984375,
        2030.508056640625
      ],
      "label": "apple",
      "score": 0.9828170537948608
    }
  ],
  "image": "......"
}
```

![image](https://github.com/user-attachments/assets/759e308e-efd4-4f78-86c8-e7003dd6e9bf)
