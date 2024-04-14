#ifndef YOLOV8_ONNXRUNTIME_TYPES_H
#define YOLOV8_ONNXRUNTIME_TYPES_H

#include <opencv2/core/types.hpp>
namespace yolov8_onnxruntime
{
/**
 * @brief Represents the results of YOLO prediction.
 *
 * This structure stores information about a detected object, including its class index,
 * confidence score, bounding box, semantic segmentation mask, and keypoints (if available).
 */
struct YoloResults
{
  int class_idx{};                ///< The class index of the detected object.
  float conf{};                   ///< The confidence score of the detection.
  cv::Rect_<float> bbox;          ///< The bounding box of the detected object.
  cv::Mat mask;                   ///< The semantic segmentation mask (if available).
  std::vector<float> keypoints{}; ///< Keypoints representing the object's pose (if available).
};

struct ImageInfo
{
  cv::Size raw_size; // add additional attrs if you need
};

} // namespace yolov8_onnxruntime

#endif // YOLOV8_ONNXRUNTIME_TYPES_H