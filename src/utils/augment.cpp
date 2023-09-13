#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>


/**
 * \brief padding value when letterbox changes image size ratio
 */
const int& DEFAULT_LETTERBOX_PAD_VALUE = 114;


void letterbox(const cv::Mat& image,
    cv::Mat& outImage,
    const cv::Size& newShape,
    cv::Scalar_<double> color,
    bool auto_,
    bool scaleFill,
    bool scaleUp, int stride
) {
    cv::Size shape = image.size();
    float r = std::min(static_cast<float>(newShape.height) / static_cast<float>(shape.height),
        static_cast<float>(newShape.width) / static_cast<float>(shape.width));
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int newUnpad[2]{ static_cast<int>(std::round(static_cast<float>(shape.width) * r)),
                     static_cast<int>(std::round(static_cast<float>(shape.height) * r)) };

    auto dw = static_cast<float>(newShape.width - newUnpad[0]);
    auto dh = static_cast<float>(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = static_cast<float>((static_cast<int>(dw) % stride));
        dh = static_cast<float>((static_cast<int>(dh) % stride));
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = static_cast<float>(newShape.width) / static_cast<float>(shape.width);
        ratio[1] = static_cast<float>(newShape.height) / static_cast<float>(shape.height);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    //cv::Mat outImage;
    if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else
    {
        outImage = image.clone();
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));


    if (color == cv::Scalar()) {
        color = cv::Scalar(DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE);
    }

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

}

cv::Mat scale_image(const cv::Mat& resized_mask, const cv::Size& im0_shape, const std::pair<float, cv::Point2f>& ratio_pad) {
    cv::Size im1_shape = resized_mask.size();

    // Check if resizing is needed
    if (im1_shape == im0_shape) {
        return resized_mask.clone();
    }

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(im1_shape.height) / static_cast<float>(im0_shape.height),
            static_cast<float>(im1_shape.width) / static_cast<float>(im0_shape.width));
        pad_x = (im1_shape.width - im0_shape.width * gain) / 2.0f;
        pad_y = (im1_shape.height - im0_shape.height * gain) / 2.0f;
    }
    else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    int top = static_cast<int>(pad_y);
    int left = static_cast<int>(pad_x);
    int bottom = static_cast<int>(im1_shape.height - pad_y);
    int right = static_cast<int>(im1_shape.width - pad_x);

    // Clip and resize the mask
    cv::Rect clipped_rect(left, top, right - left, bottom - top);
    cv::Mat clipped_mask = resized_mask(clipped_rect);
    cv::Mat scaled_mask;
    cv::resize(clipped_mask, scaled_mask, im0_shape);

    return scaled_mask;
}


void scale_image2(cv::Mat& scaled_mask, const cv::Mat& resized_mask, const cv::Size& im0_shape, const std::pair<float, cv::Point2f>& ratio_pad) {
    cv::Size im1_shape = resized_mask.size();

    // Check if resizing is needed
    if (im1_shape == im0_shape) {
        scaled_mask = resized_mask.clone();
        return;
    }

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(im1_shape.height) / static_cast<float>(im0_shape.height),
                        static_cast<float>(im1_shape.width) / static_cast<float>(im0_shape.width));
        pad_x = (im1_shape.width - im0_shape.width * gain) / 2.0f;
        pad_y = (im1_shape.height - im0_shape.height * gain) / 2.0f;
    }
    else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    int top = static_cast<int>(pad_y);
    int left = static_cast<int>(pad_x);
    int bottom = static_cast<int>(im1_shape.height - pad_y);
    int right = static_cast<int>(im1_shape.width - pad_x);

    // Clip and resize the mask
    cv::Rect clipped_rect(left, top, right - left, bottom - top);
    cv::Mat clipped_mask = resized_mask(clipped_rect);
    cv::resize(clipped_mask, scaled_mask, im0_shape);
}
