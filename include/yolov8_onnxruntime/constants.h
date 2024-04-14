#ifndef YOLOV8_ONNXRUNTIME_CONSTANTS_H
#define YOLOV8_ONNXRUNTIME_CONSTANTS_H
#include <string>
namespace yolov8_onnxruntime
{
namespace MetadataConstants
{
inline const std::string IMGSZ = "imgsz";
inline const std::string STRIDE = "stride";
inline const std::string NC = "nc";
inline const std::string CH = "ch";
inline const std::string DATE = "date";
inline const std::string VERSION = "version";
inline const std::string TASK = "task";
inline const std::string BATCH = "batch";
inline const std::string NAMES = "names";
} // namespace MetadataConstants

enum class OnnxProviders_t
{
  CPU,
  CUDA,
  OPENVINO
};
namespace OnnxProviders
{
inline const std::string CPU = "cpu";
inline const std::string CUDA = "cuda";
inline const std::string OPENVINO = "openvino";
} // namespace OnnxProviders

inline constexpr std::string_view OnnxProviderToString(const OnnxProviders_t provider)
{
  switch (provider)
  {
  case OnnxProviders_t::CPU:
    return OnnxProviders::CPU;
  case OnnxProviders_t::CUDA:
    return OnnxProviders::CUDA;
  case OnnxProviders_t::OPENVINO:
    return OnnxProviders::OPENVINO;
  default:
    return OnnxProviders::CPU;
  }
}

namespace OnnxInitializers
{
inline const int UNINITIALIZED_STRIDE = -1;
inline const int UNINITIALIZED_NC = -1;
} // namespace OnnxInitializers

namespace YoloTasks
{
inline const std::string SEGMENT = "segment";
inline const std::string DETECT = "detect";
inline const std::string POSE = "pose";
inline const std::string CLASSIFY = "classify";
} // namespace YoloTasks
} // namespace yolov8_onnxruntime

#endif // YOLOV8_ONNXRUNTIME_CONSTANTS_H