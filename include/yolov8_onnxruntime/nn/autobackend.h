#ifndef YOLOV8_ONNXRUNTIME_AUTOBACKEND_H
#define YOLOV8_ONNXRUNTIME_AUTOBACKEND_H
#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <unordered_map>
#include <vector>

#include "yolov8_onnxruntime/constants.h"
#include "yolov8_onnxruntime/nn/onnx_model_base.h"

#include "yolov8_onnxruntime/types.h"

namespace yolov8_onnxruntime
{

class AutoBackendOnnx : public OnnxModelBase
{
public:
  // constructors
  AutoBackendOnnx(const char* modelPath,
                  const char* logid,
                  const OnnxProviders_t provider,
                  const std::vector<int>& imgsz,
                  const int& stride,
                  const int& nc,
                  std::unordered_map<int, std::string> names);

  AutoBackendOnnx(const char* modelPath, const char* logid, const OnnxProviders_t provider);

  // getters
  virtual const std::vector<int>& getImgsz();
  virtual const int& getStride();
  virtual const int& getCh();
  virtual const int& getNc();
  virtual const std::unordered_map<int, std::string>& getNames();
  virtual const std::vector<int64_t>& getInputTensorShape();
  virtual const int& getWidth();
  virtual const int& getHeight();
  virtual const cv::Size& getCvSize();
  virtual const std::string& getTask();

  std::vector<std::vector<std::vector<cv::Point>>>
  getBoundaryPoints(const std::vector<YoloResults>& objs) const;

  /**
   * @brief Runs object detection on an input image.
   *
   * This method performs object detection on the input image and returns the detected objects as
   * YoloResults.
   *
   * @param image The input image to run object detection on.
   * @param conf The confidence threshold for object detection.
   * @param iou The intersection-over-union (IoU) threshold for non-maximum suppression.
   * @param mask_threshold The threshold for the semantic segmentation mask.
   * @param conversionCode An optional conversion code for image format conversion (e.g.,
   * cv::COLOR_BGR2RGB). Default value is -1, indicating no conversion.
   *                      TODO: use some constant from some namespace rather than hardcoded values
   * here
   *
   * @return A vector of YoloResults representing the detected objects.
   */
  virtual std::vector<YoloResults> predict_once(cv::Mat& image,
                                                float& conf,
                                                float& iou,
                                                float& mask_threshold,
                                                int conversionCode = -1,
                                                bool verbose = true);
  virtual std::vector<YoloResults> predict_once(const std::filesystem::path& imagePath,
                                                float& conf,
                                                float& iou,
                                                float& mask_threshold,
                                                int conversionCode = -1,
                                                bool verbose = true);
  virtual std::vector<YoloResults> predict_once(const std::string& imagePath,
                                                float& conf,
                                                float& iou,
                                                float& mask_threshold,
                                                int conversionCode = -1,
                                                bool verbose = true);

  virtual void fill_blob(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
  virtual void postprocess_masks(cv::Mat& output0,
                                 cv::Mat& output1,
                                 ImageInfo para,
                                 std::vector<YoloResults>& output,
                                 int& class_names_num,
                                 float& conf_threshold,
                                 float& iou_threshold,
                                 int& iw,
                                 int& ih,
                                 int& mw,
                                 int& mh,
                                 int& masks_features_num,
                                 float mask_threshold = 0.50f);

  virtual void postprocess_detects(cv::Mat& output0,
                                   ImageInfo image_info,
                                   std::vector<YoloResults>& output,
                                   int& class_names_num,
                                   float& conf_threshold,
                                   float& iou_threshold);
  virtual void postprocess_kpts(cv::Mat& output0,
                                ImageInfo& image_info,
                                std::vector<YoloResults>& output,
                                int& class_names_num,
                                float& conf_threshold,
                                float& iou_threshold);
  static void _get_mask2(const cv::Mat& mask_info,
                         const cv::Mat& mask_data,
                         const ImageInfo& image_info,
                         cv::Rect bound,
                         cv::Mat& mask_out,
                         float& mask_thresh,
                         int& iw,
                         int& ih,
                         int& mw,
                         int& mh,
                         int& masks_features_num,
                         bool round_downsampled = false);

protected:
  std::vector<int> imgsz_;
  int stride_ = OnnxInitializers::UNINITIALIZED_STRIDE;
  int nc_ = OnnxInitializers::UNINITIALIZED_NC; //
  int ch_ = 3;
  std::unordered_map<int, std::string> names_;
  std::vector<int64_t> inputTensorShape_;
  cv::Size cvSize_;
  std::string task_;
  // cv::MatSize cvMatSize_;
};

} // namespace yolov8_onnxruntime

#endif // YOLOV8_ONNXRUNTIME_AUTOBACKEND_H