#pragma once

#include "nn/autobackend.h"

#include <iostream>
#include <ostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

#include "utils/augment.h"
#include "constants.h"
#include "utils/common.h"
#include "utils/ops.h"


namespace fs = std::filesystem;


AutoBackendOnnx::AutoBackendOnnx(const char* modelPath, const char* logid, const char* provider,
    const std::vector<int>& imgsz, const int& stride,
    const int& nc, const std::unordered_map<int, std::string> names)
    : OnnxModelBase(modelPath, logid, provider), imgsz_(imgsz), stride_(stride), nc_(nc), names_(names),
    inputTensorShape_()
{
}

AutoBackendOnnx::AutoBackendOnnx(const char* modelPath, const char* logid, const char* provider)
    : OnnxModelBase(modelPath, logid, provider) {
    // init metadata etc
    OnnxModelBase(modelPath, logid, provider);
    // then try to get additional info from metadata like imgsz, stride etc;
    //  ideally you should get all of them but you'll raise error if smth is not in metadata (or not under the appropriate keys)
    const std::unordered_map<std::string, std::string>& base_metadata = OnnxModelBase::getMetadata();

    // post init imgsz
    auto imgsz_iterator = base_metadata.find(MetadataConstants::IMGSZ);
    if (imgsz_iterator != base_metadata.end()) {
        // parse it and convert to int iterable
        std::vector<int> imgsz = convertStringVectorToInts(parseVectorString(imgsz_iterator->second));
        // set it here:
        if (imgsz_.empty()) {
            imgsz_ = imgsz;
        }
    }
    else {
        std::cerr << "Warning: Cannot get imgsz value from metadata" << std::endl;
    }

    // post init stride
    auto stride_item = base_metadata.find(MetadataConstants::STRIDE);
    if (stride_item != base_metadata.end()) {
        // parse it and convert to int iterable
        int stide_int = std::stoi(stride_item->second);
        // set it here:
        if (stride_ == OnnxInitializers::UNINITIALIZED_STRIDE) {
            stride_ = stide_int;
        }
    }
    else {
        std::cerr << "Warning: Cannot get stride value from metadata" << std::endl;
    }

    // post init names
    auto names_item = base_metadata.find(MetadataConstants::NAMES);
    if (names_item != base_metadata.end()) {
        // parse it and convert to int iterable
        std::unordered_map<int, std::string> names = parseNames(names_item->second);
        std::cout << "***Names from metadata***" << std::endl;
        for (const auto& pair : names) {
            std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        }
        // set it here:
        if (names_.empty()) {
            names_ = names;
        }
    }
    else {
        std::cerr << "Warning: Cannot get names value from metadata" << std::endl;
    }

    // post init number of classes - you can do that only and only if names_ is not empty and nc was not initialized previously
    if (nc_ == OnnxInitializers::UNINITIALIZED_NC && !names_.empty()) {
        nc_ = names_.size();
    }
    else {
        std::cerr << "Warning: Cannot get nc value from metadata (probably names wasn't set)" << std::endl;
    }

    if (!imgsz_.empty() && inputTensorShape_.empty())
    {
        inputTensorShape_ = { 1, ch_, getHeight(), getWidth() };
    }

    if (!imgsz_.empty())
    {
        // Initialize cvSize_ using getHeight() and getWidth()
        //cvSize_ = cv::MatSize()
        cvSize_ = cv::Size(getWidth(), getHeight());
        //cvMatSize_ = cv::MatSize(cvSize_.width, cvSize_.height);
    }

    // task init:
    auto task_item = base_metadata.find(MetadataConstants::TASK);
    if (task_item != base_metadata.end()) {
        // parse it and convert to int iterable
        std::string task = std::string(task_item->second);
        // set it here:
        if (task_.empty())
        {
            task_ = task;
        }
    }
    else {
        std::cerr << "Warning: Cannot get task value from metadata" << std::endl;
    }

    // TODO: raise assert if imgsz_ and task_ were not initialized (since you don't know in that case which postprocessing to use)

}



const std::vector<int>& AutoBackendOnnx::getImgsz() {
    return imgsz_;
}

const int& AutoBackendOnnx::getHeight()
{
    return imgsz_[0];
}

const int& AutoBackendOnnx::getWidth()
{
    return imgsz_[1];
}

const int& AutoBackendOnnx::getStride() {
    return stride_;
}

const int& AutoBackendOnnx::getCh() {
    return ch_;
}

const int& AutoBackendOnnx::getNc() {
    return nc_;
}

const std::unordered_map<int, std::string>& AutoBackendOnnx::getNames() {
    return names_;
}


const cv::Size& AutoBackendOnnx::getCvSize()
{
    return cvSize_;
}

const std::vector<int64_t>& AutoBackendOnnx::getInputTensorShape()
{
    return inputTensorShape_;
}

const std::string& AutoBackendOnnx::getTask()
{
    return task_;
}

std::vector<YoloResults> AutoBackendOnnx::predict_once(const std::string& imagePath, float& conf, float& iou, float& mask_threshold,
    int conversionCode, bool verbose) {
    // Convert the string imagePath to an object of type std::filesystem::path
    fs::path imageFilePath(imagePath);
    // Call the predict_once method, converting the image to a cv::Mat
    return predict_once(imageFilePath, conf, iou, mask_threshold, conversionCode);
}

std::vector<YoloResults> AutoBackendOnnx::predict_once(const fs::path& imagePath, float& conf, float& iou, float& mask_threshold,
    int conversionCode, bool verbose) {
    // Check if the specified path exists
    if (!fs::exists(imagePath)) {
        std::cerr << "Error: File does not exist: " << imagePath << std::endl;
        // Return an empty vector or throw an exception, depending on your logic
        return std::vector<YoloResults>();
    }

    // Load the image into a cv::Mat
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_UNCHANGED);

    // Check if loading the image was successful
    if (image.empty()) {
        std::cerr << "Error: Failed to load image: " << imagePath << std::endl;
        // Return an empty vector or throw an exception, depending on your logic
        return std::vector<YoloResults>();
    }

    // now do some preprocessing based on channels info:
    int required_image_channels = this->getCh();
    /*assert(required_image_channels == image.channels() && "");*/
    // Assert that the number of channels in the input image matches the required number of channels for the model
    if (required_image_channels != image.channels()) {
        const std::string& errorMessage = "Error: Number of image channels does not match the required channels.\n"
            "Number of channels in the image: " + std::to_string(image.channels());
        throw std::runtime_error(errorMessage);
    }

    // Call overloaded one
    return predict_once(image, conf, iou, mask_threshold, conversionCode);
}


// predict method should be similar to what we have in python - image (or it's existing path), conf, iou etc
//  additionally, we can put here mask_threshold (
//      in python after the sigmoid is applied to matrix
//      product between features mask and protos the threshold is gt_(0.5)
//      as masks.gt_(0.5) in ultralytics/utils/ops.process_mask
// )

std::vector<YoloResults> AutoBackendOnnx::predict_once(cv::Mat& image, float& conf, float& iou, float& mask_threshold, int conversionCode, bool verbose) {
    double preprocess_time = 0.0;
    double inference_time = 0.0;
    double postprocess_time = 0.0;
    Timer preprocess_timer = Timer(preprocess_time, verbose);
    // 1. preprocess
    float* blob = nullptr;
    //double* blob = nullptr;
    std::vector<Ort::Value> inputTensors;
    if (conversionCode >= 0) {
        cv::cvtColor(image, image, conversionCode);
    }
    std::vector<int64_t> inputTensorShape;
    // TODO: for classify task preprocessed image will be different (!):
    cv::Mat preprocessed_img;
    cv::Size new_shape = cv::Size(getWidth(), getHeight());
    const bool& scaleFill = false;  // false
    const bool& auto_ = false; // true
    letterbox(image, preprocessed_img, new_shape, cv::Scalar(), auto_, scaleFill, true, getStride());
    fill_blob(preprocessed_img, blob, inputTensorShape);
    int64_t inputTensorSize = vector_product(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()
    ));
    preprocess_timer.Stop();
    Timer inference_timer = Timer(inference_time, verbose);
    // 2. inference
    std::vector<Ort::Value> outputTensors = forward(inputTensors);
    inference_timer.Stop();
    Timer postprocess_timer = Timer(postprocess_time, verbose);
    // create container for the results
    std::vector<YoloResults> results;
    // 3. postprocess based on task:
    std::unordered_map<int, std::string> names = this->getNames();
    int class_names_num = names.size();
    if (task_ == YoloTasks::SEGMENT) {

        // get outputs info
        std::vector<int64_t> outputTensor0Shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> outputTensor1Shape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        // get outputs
        float* all_data0 = outputTensors[0].GetTensorMutableData<float>();

        cv::Mat output0 = cv::Mat(cv::Size((int)outputTensor0Shape[2], (int)outputTensor0Shape[1]), CV_32F, all_data0).t();  // [bs, features, preds_num]=>[bs, preds_num, features]
        auto mask_shape = outputTensor1Shape;
        std::vector<int> mask_sz = { 1,(int)mask_shape[1],(int)mask_shape[2],(int)mask_shape[3] };
        cv::Mat output1 = cv::Mat(mask_sz, CV_32F, outputTensors[1].GetTensorMutableData<float>());

        int iw = this->getWidth();
        int ih = this->getHeight();
        int mask_features_num = outputTensor1Shape[1];
        int mh = outputTensor1Shape[2];
        int mw = outputTensor1Shape[3];
        ImageInfo img_info = { image.size() };
        postprocess_masks(output0, output1, img_info, results, class_names_num, conf, iou,
            iw, ih, mw, mh, mask_features_num, mask_threshold);
    }
    else if (task_ == YoloTasks::DETECT) {
        ImageInfo img_info = { image.size() };
        std::vector<int64_t> outputTensor0Shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        float* all_data0 = outputTensors[0].GetTensorMutableData<float>();
        cv::Mat output0 = cv::Mat(cv::Size((int)outputTensor0Shape[2], (int)outputTensor0Shape[1]), CV_32F, all_data0).t();  // [bs, features, preds_num]=>[bs, preds_num, features]
        postprocess_detects(output0, img_info, results, class_names_num, conf, iou);
    }
    else {
        throw std::runtime_error("NotImplementedError: task: " + task_);
    }

    postprocess_timer.Stop();
    if (verbose) {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "image: " << preprocessed_img.rows << "x" << preprocessed_img.cols << " " << results.size() << " objs, ";
        std::cout << (preprocess_time + inference_time + postprocess_time) * 1000.0 << "ms" << std::endl;
        std::cout << "Speed: " << (preprocess_time * 1000.0) << "ms preprocess, ";
        std::cout << (inference_time * 1000.0) << "ms inference, ";
        std::cout << (postprocess_time * 1000.0) << "ms postprocess per image ";
        std::cout << "at shape (1, " << image.channels() << ", " << preprocessed_img.rows << ", " << preprocessed_img.cols << ")" << std::endl;
    }

    return results;
}


void AutoBackendOnnx::postprocess_masks(cv::Mat& output0, cv::Mat& output1, ImageInfo image_info, std::vector<YoloResults>& output,
    int& class_names_num, float& conf_threshold, float& iou_threshold,
    int& iw, int& ih, int& mw, int& mh, int& masks_features_num, float mask_threshold /* = 0.5f */)
{
    output.clear();
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> masks;
    // 4 - your default number of rect parameters {x, y, w, h}
    int data_width = class_names_num + 4 + masks_features_num;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;
    // for loop original source: https://github.com/winxos/yolov8_segment_onnx_in_cpp/blob/ab6a1db14faa932cf9ac2dbef031f9b50c658ac5/onnxcpp/main.cpp
    for (int r = 0; r < rows; ++r)
    {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        minMaxLoc(scores, 0, &max_conf, 0, &class_id);
        // ultralytics/utils/ops.non_max_suppression line 206: 
        //     xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
        // so there's must be strict condition
        if (max_conf > conf_threshold)
        {
            masks.push_back(std::vector<float>(pdata + 4 + class_names_num, pdata + data_width));
            class_ids.push_back(class_id.x);
            confidences.push_back(max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
            float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);
            cv::Rect_ <float> bbox = cv::Rect(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            cv::Rect_<float> scaled_bbox = scale_boxes(getCvSize(), bbox, image_info.raw_size);
            boxes.push_back(scaled_bbox);
        }
        pdata += data_width; // next pred
    }

    // 
    //float masks_threshold = 0.50;
    //int top_k = 500;
    //const float& nmsde_eta = 1.0f;
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result); // , nms_eta, top_k);
    for (int i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        boxes[idx] = boxes[idx] & cv::Rect(0, 0, image_info.raw_size.width, image_info.raw_size.height);
        YoloResults result = { class_ids[idx] ,confidences[idx] ,boxes[idx] };
        _get_mask2(cv::Mat(masks[idx]).t(), output1, image_info, boxes[idx], result.mask, mask_threshold,
            iw, ih, mw, mh, masks_features_num);
        output.push_back(result);
    }
}


void AutoBackendOnnx::postprocess_detects(cv::Mat& output0, ImageInfo image_info, std::vector<YoloResults>& output,
    int& class_names_num, float& conf_threshold, float& iou_threshold)
{
    output.clear();
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> masks;
    // 4 - your default number of rect parameters {x, y, w, h}
    int data_width = class_names_num + 4;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;

    for (int r = 0; r < rows; ++r)
    {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        minMaxLoc(scores, 0, &max_conf, 0, &class_id);

        if (max_conf > conf_threshold)
        {
            masks.push_back(std::vector<float>(pdata + 4 + class_names_num, pdata + data_width));
            class_ids.push_back(class_id.x);
            confidences.push_back(max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
            float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);

            cv::Rect_ <float> bbox = cv::Rect(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            cv::Rect_<float> scaled_bbox = scale_boxes(getCvSize(), bbox, image_info.raw_size);

            boxes.push_back(scaled_bbox);
        }
        pdata += data_width; // next pred
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result); // , nms_eta, top_k);
    for (int i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        boxes[idx] = boxes[idx] & cv::Rect(0, 0, image_info.raw_size.width, image_info.raw_size.height);
        YoloResults result = { class_ids[idx] ,confidences[idx] ,boxes[idx] };
        output.push_back(result);
    }
}

void AutoBackendOnnx::_get_mask2(const cv::Mat& masks_features, const cv::Mat& mask_data,
    const ImageInfo& image_info, const cv::Rect bound, cv::Mat& mask_out,
    float& mask_thresh, int& iw, int& ih, int& mw, int& mh, int& masks_features_num, bool round_downsampled,
    bool adjust_with_padding)

{
    cv::Size img0_shape = image_info.raw_size;
    cv::Size img1_shape = cv::Size(iw, ih);
    cv::Size downsampled_size = cv::Size(mw, mh);
    cv::Size bbox_size = cv::Size(bound.width, bound.height);

    cv::Rect_<float> bound_float(
        static_cast<float>(bound.x),
        static_cast<float>(bound.y),
        static_cast<float>(bound.width),
        static_cast<float>(bound.height)
    );

    cv::Rect_<float> downsampled_bbox = scale_boxes(img0_shape, bound_float, downsampled_size);
    cv::Size bound_size = cv::Size(mw, mh);
    clip_boxes(downsampled_bbox, bound_size);

    // TODO: make round vs strict casting optional parameter since it may affect results (!)
    cv::Rect downsampled_bbox_int;
    if (round_downsampled)
    {
        cv::Rect downsampled_bbox_int(
            cvRound(downsampled_bbox.x),
            cvRound(downsampled_bbox.y),
            cvRound(downsampled_bbox.width),
            cvRound(downsampled_bbox.height)
        );
    }
    else
    {
        cv::Rect downsampled_bbox_int(
            static_cast<int>(downsampled_bbox.x),
            static_cast<int>(downsampled_bbox.y),
            static_cast<int>(downsampled_bbox.width),
            static_cast<int>(downsampled_bbox.height)
        );
    }
    // select all of the protos tensor
    std::vector<cv::Range> roi_rangs = { cv::Range(0, 1), cv::Range::all(),
        cv::Range(0, downsampled_size.height), cv::Range(0, downsampled_size.width) };
    cv::Mat temp_mask = mask_data(roi_rangs).clone();
    cv::Mat proto = temp_mask.reshape(0, { masks_features_num, downsampled_size.width * downsampled_size.height });
    // perform mm between mask features and proto
    cv::Mat matmul_res = (masks_features * proto).t();
    matmul_res = matmul_res.reshape(1, { downsampled_size.height, downsampled_size.width });
    // apply sigmoid to the mask:
    cv::Mat sigmoid_mask;
    exp(-matmul_res, sigmoid_mask);
    sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
    //cv::resize(resized_mask_as_input, resized_mask, upsample_size); // , 0, 0, cv::INTER_LINEAR);
    // somehow interpolating directly from sigmoid_mask(downsampled_bbox) into original size results in less accurate mask
    //  so first create resized mask which will be the one as the input_image size and is almost identical to what we have in python
    cv::Mat resized_mask;
    cv::Rect_<float> input_bbox = scale_boxes(img0_shape, bound_float, img1_shape);
    cv::resize(sigmoid_mask, resized_mask, img1_shape); // , 0, 0, cv::INTER_LINEAR);
    cv::Mat pre_out_mask = resized_mask(input_bbox);
    cv::Mat scaled_mask = scale_image(resized_mask, img0_shape);
    cv::resize(scaled_mask, mask_out, img0_shape);
    mask_out = mask_out(bound) > mask_thresh;
}


void AutoBackendOnnx::fill_blob(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape) {

	cv::Mat floatImage;
    if (inputTensorShape.empty())
    {
        inputTensorShape = getInputTensorShape();
    }
    int inputChannelsNum = inputTensorShape[1];
    int rtype = CV_32FC3;
    image.convertTo(floatImage, rtype, 1.0f / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

