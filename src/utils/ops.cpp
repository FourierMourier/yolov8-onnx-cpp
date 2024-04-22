#pragma once

#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <vector>



void clip_boxes(cv::Rect& box, const cv::Size& shape) {
    box.x = std::max(0, std::min(box.x, shape.width));
    box.y = std::max(0, std::min(box.y, shape.height));
    box.width = std::max(0, std::min(box.width, shape.width - box.x));
    box.height = std::max(0, std::min(box.height, shape.height - box.y));
}

void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape) {
    box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
    box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
    box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
    box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
}


void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape) {
    for (cv::Rect& box : boxes) {
        clip_boxes(box, shape);
    }
}

void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape) {
    for (cv::Rect_<float>& box : boxes) {
        clip_boxes(box, shape);
    }
}

// source: ultralytics/utils/ops.py scale_boxes lines 99+ (ultralytics==8.0.160)
cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape,
    std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true) {

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
            static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
        pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
        pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
    }
    else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    //cv::Rect scaledCoords(box);
    cv::Rect_<float> scaledCoords(box);

    if (padding) {
        scaledCoords.x -= pad_x;
        scaledCoords.y -= pad_y;
    }

    scaledCoords.x /= gain;
    scaledCoords.y /= gain;
    scaledCoords.width /= gain;
    scaledCoords.height /= gain;

    // Clip the box to the bounds of the image
    clip_boxes(scaledCoords, img0_shape);

    return scaledCoords;
}


//void clip_coords(cv::Mat& coords, const cv::Size& shape) {
//    // Clip x-coordinates to the image width
//    cv::Mat xCoords = coords.col(0);
//    cv::Mat yCoords = coords.col(1);
//
//    for (int i = 0; i < coords.rows; ++i) {
//        xCoords.at<float>(i) = std::max(std::min(xCoords.at<float>(i), static_cast<float>(shape.width - 1)), 0.0f);
//        yCoords.at<float>(i) = std::max(std::min(yCoords.at<float>(i), static_cast<float>(shape.height - 1)), 0.0f);
//    }
//}

void clip_coords(std::vector<float>& coords, const cv::Size& shape) {
    // Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < coords.size(); i += 3) {
        coords[i] = std::min(std::max(coords[i], 0.0f), static_cast<float>(shape.width - 1));  // x
        coords[i + 1] = std::min(std::max(coords[i + 1], 0.0f), static_cast<float>(shape.height - 1));  // y
    }
}

// source: ultralytics/utils/ops.py scale_coords lines 753+ (ultralytics==8.0.160)
//cv::Mat scale_coords(const cv::Size& img1_shape, cv::Mat& coords, const cv::Size& img0_shape)
//cv::Mat scale_coords(const cv::Size& img1_shape, std::vector<float> coords, const cv::Size& img0_shape)
std::vector<float> scale_coords(const cv::Size& img1_shape, std::vector<float>& coords, const cv::Size& img0_shape)
{
//    cv::Mat scaledCoords = coords.clone();
    std::vector<float> scaledCoords = coords;

    // Calculate gain and padding
    double gain = std::min(static_cast<double>(img1_shape.width) / img0_shape.width, static_cast<double>(img1_shape.height) / img0_shape.height);
    cv::Point2d pad((img1_shape.width - img0_shape.width * gain) / 2, (img1_shape.height - img0_shape.height * gain) / 2);

    // Apply padding
//    scaledCoords.col(0) = (scaledCoords.col(0) - pad.x);
//    scaledCoords.col(1) = (scaledCoords.col(1) - pad.y);
    // Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i] -= pad.x;  // x padding
        scaledCoords[i + 1] -= pad.y;  // y padding
    }

    // Scale coordinates
//    scaledCoords.col(0) /= gain;
//    scaledCoords.col(1) /= gain;
    // Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i] /= gain;
        scaledCoords[i + 1] /= gain;
    }

    clip_coords(scaledCoords, img0_shape);
    return scaledCoords;
}


cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect& box) {
    int h = mask.rows;
    int w = mask.cols;

    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.x + box.width;
    int y2 = box.y + box.height;

    cv::Mat cropped_mask = cv::Mat::zeros(h, w, mask.type());

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            if (r >= y1 && r < y2 && c >= x1 && c < x2) {
                cropped_mask.at<float>(r, c) = mask.at<float>(r, c);
            }
        }
    }

    return cropped_mask;
}

//std::tuple<std::vector<cv::Rect_<float>>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
non_max_suppression(const cv::Mat& output0, int class_names_num, int data_width, double conf_threshold,
                    float iou_threshold) {

    std::vector<int> class_ids;
    std::vector<float> confidences;
//    std::vector<cv::Rect_<float>> boxes;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> rest;

    int rest_start_pos = class_names_num + 4;
    int rest_features = data_width - rest_start_pos;
//    int data_width = rest_start_pos + total_features_num;

    int rows = output0.rows;
    float* pdata = (float*) output0.data;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);

        if (max_conf > conf_threshold) {
            std::vector<float> mask_data(pdata + 4 + class_names_num, pdata + data_width);
            class_ids.push_back(class_id.x);
            confidences.push_back((float) max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
            float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);
            cv::Rect_<float> bbox(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            boxes.push_back(bbox);
            if (rest_features > 0) {
                std::vector<float> rest_data(pdata + rest_start_pos, pdata + data_width);
                rest.push_back(rest_data);
            }
        }
        pdata += data_width; // next prediction
    }

    //
    //float masks_threshold = 0.50;
    //int top_k = 500;
    //const float& nmsde_eta = 1.0f;
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result); // , nms_eta, top_k);
//    cv::dnn::NMSBoxes(boxes, confidences, );
    std::vector<int> nms_class_ids;
    std::vector<float> nms_confidences;
//    std::vector<cv::Rect_<float>> boxes;
    std::vector<cv::Rect> nms_boxes;
    std::vector<std::vector<float>> nms_rest;
    for (int idx: nms_result) {
        nms_class_ids.push_back(class_ids[idx]);
        nms_confidences.push_back(confidences[idx]);
        nms_boxes.push_back(boxes[idx]);
        nms_rest.push_back(rest[idx]);
    }
    return std::make_tuple(nms_boxes, nms_confidences, nms_class_ids, nms_rest);
}
