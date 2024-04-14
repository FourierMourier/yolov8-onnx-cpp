
#include <opencv2/core.hpp>
#include <yolov8_onnxruntime/nn/autobackend.h>

#include <yolov8_onnxruntime/utils/viz.h>

#include <iostream>

// #include <opencv2/core/mat.hpp>
// get imread
#include <opencv2/highgui.hpp>

// #include <utils/viz.h>

int main()
{
  std::string img_path = "../images/trailer.jpg";
  // const std::img_path& modelPath = "./checkpoints/yolov8n.onnx"; // detection
  // vs:
  //    const std::string& modelPath = "./checkpoints/yolov8n-seg.onnx"; // instance segmentation
  // clion:
  const std::string& modelPath =
      "../models/trailer_opening_segmentation_v1_yolo8n-seg.onnx"; // pose

  std::filesystem::path imageFilePath(img_path);
  std::filesystem::path newFilePath = imageFilePath.stem();
  newFilePath += "-kpt-cpp";
  newFilePath += imageFilePath.extension();
  assert(newFilePath != imageFilePath);
  std::cout << "newFilePath: " << newFilePath << std::endl;

  const auto onnx_provider = yolov8_onnxruntime::OnnxProviders_t::CPU; // "cpu";
  const std::string& onnx_logid = "yolov8_inference2";
  float mask_threshold =
      0.5f; // in python it's 0.5 and you can see that at ultralytics/utils/ops.process_mask line
            // 705 (ultralytics.__version__ == .160)
  float conf_threshold = 0.30f;
  float iou_threshold = 0.45f; //  0.70f;
  int conversion_code = cv::COLOR_BGR2RGB;
  cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  if (img.empty())
  {
    std::cerr << "Error: Unable to load image" << std::endl;
    return 1;
  }
  yolov8_onnxruntime::AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider);
  std::vector<yolov8_onnxruntime::YoloResults> objs =
      model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
  std::vector<cv::Scalar> colors =
      yolov8_onnxruntime::generateRandomColors(model.getNc(), model.getCh());
  std::unordered_map<int, std::string> names = model.getNames();

  std::vector<std::vector<float>> keypointsVector;
  for (const yolov8_onnxruntime::YoloResults& result : objs)
  {
    keypointsVector.push_back(result.keypoints);
  }

  cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
  cv::Size show_shape = img.size(); // cv::Size(1280, 720); // img.size()
                                    //   plot_results(img, objs, colors, names, show_shape);
  plot_masks(img, objs, colors, names);
  cv::imshow("img", img);
  cv::waitKey();
  return -1;
}
