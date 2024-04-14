#ifndef YOLOV8_ONNXRUNTIME_VIZ_H
#define YOLOV8_ONNXRUNTIME_VIZ_H
#include <yolov8_onnxruntime/types.h>

#include <opencv2/core.hpp>

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolov8_onnxruntime
{
// todo, change these to constexpr arrays
// Define the skeleton and color mappings
inline std::vector<std::vector<int>> skeleton = {{16, 14},
                                                 {14, 12},
                                                 {17, 15},
                                                 {15, 13},
                                                 {12, 13},
                                                 {6, 12},
                                                 {7, 13},
                                                 {6, 7},
                                                 {6, 8},
                                                 {7, 9},
                                                 {8, 10},
                                                 {9, 11},
                                                 {2, 3},
                                                 {1, 2},
                                                 {1, 3},
                                                 {2, 4},
                                                 {3, 5},
                                                 {4, 6},
                                                 {5, 7}};

inline std::vector<cv::Scalar> posePalette = {
    cv::Scalar(255, 128, 0),   cv::Scalar(255, 153, 51),  cv::Scalar(255, 178, 102),
    cv::Scalar(230, 230, 0),   cv::Scalar(255, 153, 255), cv::Scalar(153, 204, 255),
    cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255),  cv::Scalar(102, 178, 255),
    cv::Scalar(51, 153, 255),  cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102),
    cv::Scalar(255, 51, 51),   cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
    cv::Scalar(51, 255, 51),   cv::Scalar(0, 255, 0),     cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 0),     cv::Scalar(255, 255, 255)};

inline std::vector<int> limbColorIndices =
    {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};
inline std::vector<int> kptColorIndices = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};

inline cv::Scalar generateRandomColor(int numChannels)
{
  if (numChannels < 1 || numChannels > 3)
  {
    throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 255);

  cv::Scalar color;
  for (int i = 0; i < numChannels; i++)
  {
    color[i] = dis(gen); // for each channel separately generate value
  }

  return color;
}

inline std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels)
{
  std::vector<cv::Scalar> colors;
  for (int i = 0; i < class_names_num; i++)
  {
    cv::Scalar color = generateRandomColor(numChannels);
    colors.push_back(color);
  }
  return colors;
}

inline void plot_masks(cv::Mat img,
                       std::vector<YoloResults>& result,
                       std::vector<cv::Scalar> color,
                       std::unordered_map<int, std::string>& names)
{
  cv::Mat mask = img.clone();
  for (int i = 0; i < result.size(); i++)
  {
    float left, top;
    left = result[i].bbox.x;
    top = result[i].bbox.y;
    int color_num = i;
    int& class_idx = result[i].class_idx;
    rectangle(img, result[i].bbox, color[result[i].class_idx], 2);

    // try to get string value corresponding to given class_idx
    std::string class_name;
    auto it = names.find(class_idx);
    if (it != names.end())
    {
      class_name = it->second;
    }
    else
    {
      std::cerr << "Warning: class_idx not found in names for class_idx = " << class_idx
                << std::endl;
      // then convert it to string anyway
      class_name = std::to_string(class_idx);
    }

    // if (result[i].mask.rows && result[i].mask.cols > 0)
    // {
    //     mask(result[i].bbox).setTo(color[result[i].class_idx], result[i].mask);
    // }
    if (result[i].mask.rows && result[i].mask.cols > 0)
    {
      // Shade the whole mask as before
      mask(result[i].bbox).setTo(color[result[i].class_idx], result[i].mask);

      // // Find contours in the mask
      // std::vector<std::vector<cv::Point>> contours;
      // cv::findContours(result[i].mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      // // Draw the contours over the shaded area with a different color or opacity
      // cv::Scalar boundaryColor = color[result[i].class_idx] * 0.5; // Example: Darker color
      // // You can also define a completely different color here
      // cv::drawContours(
      //     mask(result[i].bbox), contours, -1, boundaryColor, 2); // Thickness of 2, adjust as
      //     needed
    }
    std::stringstream labelStream;
    labelStream << class_name << " " << std::fixed << std::setprecision(2) << result[i].conf;
    std::string label = labelStream.str();

    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::Rect rect_to_fill(
        left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
    cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
    rectangle(img, rect_to_fill, color[result[i].class_idx], -1);

    putText(
        img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
  }
  addWeighted(img, 0.6, mask, 0.4, 0, img); // add mask to src
  resize(img, img, img.size());
  // imshow("img", img);
  // cv::waitKey();
}

// void plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const
// cv::Size& shape) {
inline void
plot_keypoints(cv::Mat& image, const std::vector<YoloResults>& results, const cv::Size& shape)
{

  int radius = 5;
  bool drawLines = true;

  if (results.empty())
  {
    return;
  }

  std::vector<cv::Scalar> limbColorPalette;
  std::vector<cv::Scalar> kptColorPalette;

  for (int index : limbColorIndices)
  {
    limbColorPalette.push_back(posePalette[index]);
  }

  for (int index : kptColorIndices)
  {
    kptColorPalette.push_back(posePalette[index]);
  }

  for (const auto& res : results)
  {
    auto keypoint = res.keypoints;
    bool isPose = keypoint.size() == 51; // numKeypoints == 17 && keypoints[0].size() == 3;
    drawLines &= isPose;

    // draw points
    for (int i = 0; i < 17; i++)
    {
      int idx = i * 3;
      int x_coord = static_cast<int>(keypoint[idx]);
      int y_coord = static_cast<int>(keypoint[idx + 1]);

      if (x_coord % shape.width != 0 && y_coord % shape.height != 0)
      {
        if (keypoint.size() == 3)
        {
          float conf = keypoint[2];
          if (conf < 0.5)
          {
            continue;
          }
        }
        cv::Scalar color_k = isPose ? kptColorPalette[i]
                                    : cv::Scalar(0,
                                                 0,
                                                 255); // Default to red if not in pose mode
        cv::circle(image, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
      }
    }
    // draw lines
    if (drawLines)
    {
      for (int i = 0; i < skeleton.size(); i++)
      {
        const std::vector<int>& sk = skeleton[i];
        int idx1 = sk[0] - 1;
        int idx2 = sk[1] - 1;

        int idx1_x_pos = idx1 * 3;
        int idx2_x_pos = idx2 * 3;

        int x1 = static_cast<int>(keypoint[idx1_x_pos]);
        int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
        int x2 = static_cast<int>(keypoint[idx2_x_pos]);
        int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

        float conf1 = keypoint[idx1_x_pos + 2];
        float conf2 = keypoint[idx2_x_pos + 2];

        // Check confidence thresholds
        if (conf1 < 0.5 || conf2 < 0.5)
        {
          continue;
        }

        // Check if positions are within bounds
        if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
            x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0)
        {
          continue;
        }

        // Draw a line between keypoints
        cv::Scalar color_limb = limbColorPalette[i];
        cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
      }
    }
  }
}

inline void plot_results(cv::Mat img,
                         std::vector<YoloResults>& results,
                         std::vector<cv::Scalar> color,
                         std::unordered_map<int, std::string>& names,
                         const cv::Size& shape)
{

  cv::Mat mask = img.clone();

  int radius = 5;
  bool drawLines = true;

  auto raw_image_shape = img.size();
  std::vector<cv::Scalar> limbColorPalette;
  std::vector<cv::Scalar> kptColorPalette;

  for (int index : limbColorIndices)
  {
    limbColorPalette.push_back(posePalette[index]);
  }

  for (int index : kptColorIndices)
  {
    kptColorPalette.push_back(posePalette[index]);
  }

  for (const auto& res : results)
  {
    float left = res.bbox.x;
    float top = res.bbox.y;
    int color_num = res.class_idx;

    // Draw bounding box
    rectangle(img, res.bbox, color[res.class_idx], 2);

    // Try to get the class name corresponding to the given class_idx
    std::string class_name;
    auto it = names.find(res.class_idx);
    if (it != names.end())
    {
      class_name = it->second;
    }
    else
    {
      std::cerr << "Warning: class_idx not found in names for class_idx = " << res.class_idx
                << std::endl;
      // Then convert it to a string anyway
      class_name = std::to_string(res.class_idx);
    }

    // Draw mask if available
    if (res.mask.rows && res.mask.cols > 0)
    {
      mask(res.bbox).setTo(color[res.class_idx], res.mask);
    }

    // Create label
    std::stringstream labelStream;
    labelStream << class_name << " " << std::fixed << std::setprecision(2) << res.conf;
    std::string label = labelStream.str();

    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::Rect rect_to_fill(
        left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
    cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
    rectangle(img, rect_to_fill, color[res.class_idx], -1);
    putText(
        img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);

    // Check if keypoints are available
    if (!res.keypoints.empty())
    {
      auto keypoint = res.keypoints;
      bool isPose = keypoint.size() == 51; // numKeypoints == 17 && keypoints[0].size() == 3;
      drawLines &= isPose;

      // draw points
      for (int i = 0; i < 17; i++)
      {
        int idx = i * 3;
        int x_coord = static_cast<int>(keypoint[idx]);
        int y_coord = static_cast<int>(keypoint[idx + 1]);

        if (x_coord % raw_image_shape.width != 0 && y_coord % raw_image_shape.height != 0)
        {
          if (keypoint.size() == 3)
          {
            float conf = keypoint[2];
            if (conf < 0.5)
            {
              continue;
            }
          }
          cv::Scalar color_k = isPose ? kptColorPalette[i]
                                      : cv::Scalar(0,
                                                   0,
                                                   255); // Default to red if not in pose mode
          cv::circle(img, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
        }
      }
      // draw lines
      if (drawLines)
      {
        for (int i = 0; i < skeleton.size(); i++)
        {
          const std::vector<int>& sk = skeleton[i];
          int idx1 = sk[0] - 1;
          int idx2 = sk[1] - 1;

          int idx1_x_pos = idx1 * 3;
          int idx2_x_pos = idx2 * 3;

          int x1 = static_cast<int>(keypoint[idx1_x_pos]);
          int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
          int x2 = static_cast<int>(keypoint[idx2_x_pos]);
          int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

          float conf1 = keypoint[idx1_x_pos + 2];
          float conf2 = keypoint[idx2_x_pos + 2];

          // Check confidence thresholds
          if (conf1 < 0.5 || conf2 < 0.5)
          {
            continue;
          }

          // Check if positions are within bounds
          if (x1 % raw_image_shape.width == 0 || y1 % raw_image_shape.height == 0 || x1 < 0 ||
              y1 < 0 || x2 % raw_image_shape.width == 0 || y2 % raw_image_shape.height == 0 ||
              x2 < 0 || y2 < 0)
          {
            continue;
          }

          // Draw a line between keypoints
          cv::Scalar color_limb = limbColorPalette[i];
          cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
        }
      }
    }
  }

  // Combine the image and mask
  addWeighted(img, 0.6, mask, 0.4, 0, img);
  //    resize(img, img, img.size());
  //    resize(img, img, shape);
  //    // Show the image
  //    imshow("img", img);
  //    cv::waitKey();
}

} // namespace yolov8_onnxruntime

#endif // YOLOV8_ONNXRUNTIME_VIZ_H