#ifndef YOLOV8_ONNXRUNTIME_AUGMENT_H
#define YOLOV8_ONNXRUNTIME_AUGMENT_H
#include <opencv2/core/types.hpp>

namespace yolov8_onnxruntime
{

void letterbox(const cv::Mat& image,
               cv::Mat& outImage,
               const cv::Size& newShape = cv::Size(640, 640),
               cv::Scalar_<double> color = cv::Scalar(),
               bool auto_ = true,
               bool scaleFill = false,
               bool scaleUp = true,
               int stride = 32);

cv::Mat scale_image(const cv::Mat& resized_mask,
                    const cv::Size& im0_shape,
                    const std::pair<float, cv::Point2f>& ratio_pad =
                        std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)));

void scale_image2(cv::Mat& scaled_mask,
                  const cv::Mat& resized_mask,
                  const cv::Size& im0_shape,
                  const std::pair<float, cv::Point2f>& ratio_pad =
                      std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)));
} // namespace yolov8_onnxruntime

#endif // YOLOV8_ONNXRUNTIME_AUGMENT_H