#pragma once
#include <opencv2/core/types.hpp>

//cv::Rect scaleCoords(const cv::Size& imageShape, const cv::Rect& coords, const cv::Size& imageOriginalShape);
/**
 * Scales a bounding box from the shape of the input image to the shape of an original image.
 *
 * @param img1_shape The shape (height, width) of the input image for the model.
 * @param box The bounding box to be scaled, specified as cv::Rect_<float>.
 * @param img0_shape The shape (height, width) of the original target image.
 * @param ratio_pad An optional parameter that specifies scaling and padding factors as a pair of values.
 *	The first value (ratio) is used for scaling, and the second value (pad) is used for padding.
 *	If not provided, default values will be used.
 * @param padding An optional boolean parameter that specifies whether padding should be applied.
 *	If set to true, padding will be applied to the bounding box.
 *
 * @return A scaled bounding box specified as cv::Rect_<float>.
 *
 * This function rescales a bounding box from the shape of the input image (img1_shape) to the shape of an original image (img0_shape).
 */
cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape, std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true);
void clip_boxes(cv::Rect& box, const cv::Size& shape);
void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape);
void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape);
void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape);

//void clip_coords(cv::Mat& coords, const cv::Size& shape);
//cv::Mat scale_coords(const cv::Size& img1_shape, cv::Mat& coords, const cv::Size& img0_shape);
void clip_coords(std::vector<float>& coords, const cv::Size& shape);
std::vector<float> scale_coords(const cv::Size& img1_shape, std::vector<float>& coords, const cv::Size& img0_shape);

cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect& box);


struct NMSResult{
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> classes;
    std::vector<std::vector<float>> rest;
};

//std::tuple<std::vector<cv::Rect_<float>>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
non_max_suppression(const cv::Mat& output0, int class_names_num, int total_features_num, double conf_threshold, float iou_threshold);