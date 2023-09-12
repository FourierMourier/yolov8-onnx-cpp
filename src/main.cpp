#include <random>

#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include <opencv2/opencv.hpp>

#include "utils/augment.h"
#include "constants.h"
#include "utils/common.h"

cv::Scalar generateRandomColor(int numChannels) {
    if (numChannels < 1 || numChannels > 3) {
        throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    cv::Scalar color;
    for (int i = 0; i < numChannels; i++) {
        color[i] = dis(gen); // for each channel separately generate value
    }

    return color;
}

std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels) {
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < class_names_num; i++) {
        cv::Scalar color = generateRandomColor(numChannels);
        colors.push_back(color);
    }
    return colors;
}

// original source: https://github.com/winxos/yolov8_segment_onnx_in_cpp/blob/ab6a1db14faa932cf9ac2dbef031f9b50c658ac5/onnxcpp/main.cpp
void plot_masks(cv::Mat img, std::vector<YoloResults>& result, std::vector<cv::Scalar> color,
    std::unordered_map<int, std::string>& names)
{
    cv::Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++)
    {
        int left, top;
        left = result[i].bbox.x;
        top = result[i].bbox.y;
        int color_num = i;
        int& class_idx = result[i].class_idx;
        rectangle(img, result[i].bbox, color[result[i].class_idx], 2);

        // try to get string value corresponding to given class_idx
        std::string class_name;
        auto it = names.find(class_idx);
        if (it != names.end()) {
            class_name = it->second;
        }
        else {
            std::cerr << "Warning: class_idx not found in names for class_idx = " << class_idx << std::endl;
            // then convert it to string anyway
            class_name = std::to_string(class_idx);
        }

        if (result[i].mask.rows && result[i].mask.cols > 0)
        {
            mask(result[i].bbox).setTo(color[result[i].class_idx], result[i].mask);
        }
        std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2) << result[i].conf;
        std::string label = labelStream.str();

    	cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rect_to_fill, color[result[i].class_idx], -1);

        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    }
    addWeighted(img, 0.6, mask, 0.4, 0, img); //add mask to src
    resize(img, img, img.size());
    imshow("img", img);
    cv::waitKey();
}

int main()
{
    std::string img_path = "./images/000000000143.jpg";
    //const std::string& modelPath = "./checkpoints/yolov8n.onnx"; // detection
    const std::string& modelPath = "./checkpoints/yolov8n-seg.onnx"; // instance segmentation
    const std::string& onnx_provider = OnnxProviders::CPU; // "cpu";
    const std::string& onnx_logid = "yolov8_inference2";
    float mask_threshold = 0.5f;  // in python it's 0.5 and you can see that at ultralytics/utils/ops.process_mask line 705 (ultralytics.__version__ == .160)
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;  //  0.70f;
	int conversion_code = cv::COLOR_BGR2RGB;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return 1;
    }

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::vector<YoloResults> objs = model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
    std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());
    std::unordered_map<int, std::string> names = model.getNames();
    plot_masks(img, objs, colors, names);
    
    return -1;
}
