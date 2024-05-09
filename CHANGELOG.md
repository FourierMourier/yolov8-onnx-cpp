# Changelog

## 2024-05-09
### Fixed 🔨
* Fixed memory leak by deleting the `blob` during the `predict_once` method call:
  [PR#7](https://github.com/FourierMourier/yolov8-onnx-cpp/pull/7), by [@dusionlike](https://github.com/dusionlike)

## 2024-04-22
### Fixed 🔨
* Fixed returning not scaled coords for the keypoint task. 
  [PR#5](https://github.com/FourierMourier/yolov8-onnx-cpp/pull/5), by [@youngday](https://github.com/youngday)
* Fixed compilation issue on Linux due to the same type `model_path` arg 
  in the `Ort::Session` constructor for both Windows and Linux. 
  [PR#4](https://github.com/FourierMourier/yolov8-onnx-cpp/pull/4), 
  by [@bhavya-goyal-22](https://github.com/bhavya-goyal-22) and [FourierMourier](https://github.com/FourierMourier)
