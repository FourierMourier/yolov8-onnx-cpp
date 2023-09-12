# yolov8-onnx-cpp

## Description
Hello there! yolov8-onnx-cpp is a C++ demo implementation of the YOLOv8 model using the ONNX library.
This project is based on the YOLOv8 model by Ultralytics.
I aimed to replicate the behavior of the Python version and achieve consistent results across various image sizes.

By the way, you don't need to specify names, img_size etc while initializing the model, since we can use ONNX metadata!

When you do export in python in onnx format, the following code executes
```python
    self.metadata = {
        'description': description,
        'author': 'Ultralytics',
        'license': 'AGPL-3.0 https://ultralytics.com/license',
        'date': datetime.now().isoformat(),
        'version': __version__,
        'stride': int(max(model.stride)),
        'task': model.task,
        'batch': self.args.batch,
        'imgsz': self.imgsz,
        'names': model.names}  # model metadata
    if model.task == 'pose':
        self.metadata['kpt_shape'] = model.model[-1].kpt_shape
```

(ultralytics 8.0.160, ultralytics/engine/exporter.py lines 221-233))

We can use this parameters at least to define stride, task, names and image size as described in the schema below:

![Schema](assets/export.png)

## Supported Tasks and Hardware

| Task       | Supported |
|------------|-----------|
| Detect     | ✔️        |
| Segment    | ✔️        |
| Classify   |           |
| Pose       |           |


| Hardware   | Supported |
|------------|-----------|
| CPU        | ✔️        |
| GPU        |           |


## Comparison between Python and C++

I exported `yolov8n.pt` and `yolov8n-seg.pt` to ONNX format with an input size of [480, 640] ([height, width]). However, 
during testing, I used 2 images from the COCO128 dataset with different image sizes than the specified input. 
This difference in sizes triggered letterboxing. I maintained consistent parameters, 
setting `conf=0.3` and `iou=0.5` for both models.

Here are the comparison results:

### Segmentation
Python Result 1

![Python Result](assets/000000000143-py-seg.jpg)

C++ Result 1

![C++ Result](assets/000000000143-seg.jpg)

Python Result 2

![Python Result](assets/000000000144-py-seg.jpg)

C++ Result 2

![C++ Result](assets/000000000144-seg.jpg)

### Object detection

Python Result 1

![Python Result](assets/000000000143-py-det.jpg)

C++ Result 1

![C++ Result](assets/000000000143-det.jpg)

Python Result 2

![Python Result](assets/000000000144-py-det.jpg)

C++ Result 2

![C++ Result](assets/000000000144-det.jpg)

## Getting Started
To get started with yolov8-onnx-cpp, follow these steps:

1. Clone the repository:
    ```shell
    git clone https://github.com/FourierMourier/yolov8-onnx-cpp.git
    ```
2. Setup additional libraries:
* [opencv](https://opencv.org/releases/) (4.80+)
* onnxruntime (1.50+) (nuget package)
* boost (1.80+) (nuget package)

3. edit in the ./src/main.cpp img_path/&modelPath:
    ```cpp
    std::string img_path = "./images/000000000143.jpg";
    //const std::string& modelPath = "./checkpoints/yolov8n.onnx"; // detection
    const std::string& modelPath = "./checkpoints/yolov8n-seg.onnx"; // instance segmentation
    const std::string& onnx_provider = OnnxProviders::CPU; // "cpu";
    ```
# Usage
Provide an input image to the application, and it will perform object detection using the YOLOv8 model.
Customize the model configuration and parameters in the code as needed.

# References
* [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
* [ONNX](https://onnx.ai)
* [OpenCV](https://opencv.org)

# License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

# Acknowledgments
Some other cool repositories I found useful (and you might too):
* https://github.com/winxos/yolov8_segment_onnx_in_cpp - another project implementing yolov8 segmentation in cpp
* https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP - tensorrt impelemntation in cpp
* https://github.com/itsnine/yolov5-onnxruntime/tree/master yolov5 onnx in C++

This README was created with the assistance of OpenAI's ChatGPT (August 3 Version), a large language model.
You can learn more about it [here](https://chat.openai.com/chat)
