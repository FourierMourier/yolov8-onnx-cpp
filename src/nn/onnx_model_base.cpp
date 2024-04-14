#include "yolov8_onnxruntime/nn/onnx_model_base.h"

#include <iostream>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "yolov8_onnxruntime/constants.h"
#include "yolov8_onnxruntime/utils/common.h"

#include <algorithm>
#include <thread>

#include <ros/console.h>

namespace yolov8_onnxruntime
{

/**
 * @brief Base class for any onnx model regarding the target.
 *
 * Wraps OrtApi.
 *
 * The caller provides a model path, logid, and provider.
 *
 * See the output logs for more information on warnings/errors that occur while processing the
 * model.
 *
 * @param[in] modelPath Path to the model file.
 * @param[in] logid Log identifier.
 * @param[in] provider Provider (e.g., "CPU" or "CUDA"). (NOTE: for now only CPU is supported)
 */

OnnxModelBase::OnnxModelBase(const char* modelPath,
                             const char* logid,
                             const OnnxProviders_t provider)
    //: modelPath_(modelPath), env(std::move(env)), session(std::move(session))
    :
    modelPath_(modelPath)
{

  // TODO: too bad passing `ORT_LOGGING_LEVEL_WARNING` by default - for some cases
  //       info level would make sense too
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, logid);
  Ort::SessionOptions sessionOptions = Ort::SessionOptions();

  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable = std::find(
      availableProviders.begin(), availableProviders.end(), std::string("CUDAExecutionProvider"));
  OrtCUDAProviderOptions cudaOption;

  auto openvinoAvailable = std::find(availableProviders.begin(),
                                     availableProviders.end(),
                                     std::string("OpenVINOExecutionProvider"));
  OrtOpenVINOProviderOptions openvinoOption;
  openvinoOption.device_type = "GPU_FP32";
  openvinoOption.num_of_threads = std::thread::hardware_concurrency() - 1;
  openvinoOption.cache_dir = "/persisted_data/openvino_cache";

  if (provider == OnnxProviders_t::CUDA)
  { // strcmp(provider, OnnxProviders::CUDA.c_str()) == true strcmp(provider, "cuda") //
    // (providerStr == "cuda")
    if (cudaAvailable == availableProviders.end())
    {
      // std::cout << "CUDA is not supported by your ONNXRuntime build. Fallback to CPU." <<
      // std::endl;
      ROS_WARN_STREAM("CUDA is not supported by your ONNXRuntime build. Fallback to CPU.");
      // std::cout << "Inference device: CPU" << std::endl;
    }
    else
    {
      // std::cout << "Inference device: GPU" << std::endl;
      sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
  }

  else if (provider == OnnxProviders_t::OPENVINO)
  {
    if (openvinoAvailable == availableProviders.end())
    {
      // std::cout << "OpenVINO is not supported by your ONNXRuntime build. Fallback to CPU." <<
      // std::endl;
      ROS_WARN_STREAM("OpenVINO is not supported by your ONNXRuntime build. Fallback to CPU.");
      // std::cout << "Inference device: CPU" << std::endl;
    }
    else
    {
      sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
      // std::cout << "Inference device: OpenVINO" << std::endl;
      sessionOptions.AppendExecutionProvider_OpenVINO(openvinoOption);
    }
  }

  else if (provider == OnnxProviders_t::CPU)
  { // strcmp(provider, OnnxProviders::CPU.c_str()) == true) (providerStr == "cpu") {
    // "cpu" by default
  }
  else
  {
    throw std::runtime_error("Provider not supported (you should never see this message)");
  }

  // std::cout << "Inference device: " << std::string(provider) << std::endl;
  ROS_INFO_STREAM("Inference device: " << OnnxProviderToString(provider));
  auto modelPathW = get_win_path(modelPath);
  std::string modelPathStr(modelPathW.begin(), modelPathW.end());
  try
  {
    auto start = std::chrono::high_resolution_clock::now();
    session = Ort::Session(env, modelPathStr.c_str(), sessionOptions);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    ROS_INFO_STREAM("Model loaded in " << dur_ms << "ms");
  }
  catch (const std::exception& e)
  {
    ROS_ERROR_STREAM("Error while loading model: " << e.what());
    throw;
  }
  // session = Ort::Session(env)
  // https://github.com/microsoft/onnxruntime/issues/14157
  // std::vector<const char*> inputNodeNames; //
  // ----------------
  // init input names
  inputNodeNames;
  std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings; // <-- newly added
  Ort::AllocatorWithDefaultOptions allocator;
  auto inputNodesNum = session.GetInputCount();
  for (int i = 0; i < inputNodesNum; i++)
  {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNodeNames.push_back(inputNodeNameAllocatedStrings.back().get());
  }
  // -----------------
  // init output names
  outputNodeNames;
  auto outputNodesNum = session.GetOutputCount();
  std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings; // <-- newly added
  Ort::AllocatorWithDefaultOptions output_names_allocator;
  for (int i = 0; i < outputNodesNum; i++)
  {
    auto output_name = session.GetOutputNameAllocated(i, output_names_allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNodeNames.push_back(outputNodeNameAllocatedStrings.back().get());
  }
  // -------------------------
  // initialize model metadata
  model_metadata = session.GetModelMetadata();
  Ort::AllocatorWithDefaultOptions metadata_allocator;

  std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys =
      model_metadata.GetCustomMetadataMapKeysAllocated(metadata_allocator);
  std::vector<std::string> metadata_keys;
  metadata_keys.reserve(metadataAllocatedKeys.size());

  for (const Ort::AllocatedStringPtr& allocatedString : metadataAllocatedKeys)
  {
    metadata_keys.emplace_back(allocatedString.get());
  }

  // -------------------------
  // initialize metadata as the dict
  // even though we know exactly what metadata we intend to use
  // base onnx class should not have any ultralytics yolo-specific attributes like stride, task etc,
  // so keep it clean as much as possible
  for (const std::string& key : metadata_keys)
  {
    Ort::AllocatedStringPtr metadata_value =
        model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), metadata_allocator);
    if (metadata_value != nullptr)
    {
      auto raw_metadata_value = metadata_value.get();
      metadata[key] = std::string(raw_metadata_value);
    }
  }

  // initialize cstr
  for (const std::string& name : outputNodeNames)
  {
    outputNamesCStr.push_back(name.c_str());
  }

  for (const std::string& name : inputNodeNames)
  {
    inputNamesCStr.push_back(name.c_str());
  }
}

const std::vector<std::string>& OnnxModelBase::getInputNames() { return inputNodeNames; }

const std::vector<std::string>& OnnxModelBase::getOutputNames() { return outputNodeNames; }

const Ort::ModelMetadata& OnnxModelBase::getModelMetadata() { return model_metadata; }

const std::unordered_map<std::string, std::string>& OnnxModelBase::getMetadata()
{
  return metadata;
}

const Ort::Session& OnnxModelBase::getSession() { return session; }

const char* OnnxModelBase::getModelPath() { return modelPath_; }

const std::vector<const char*> OnnxModelBase::getOutputNamesCStr() { return outputNamesCStr; }

const std::vector<const char*> OnnxModelBase::getInputNamesCStr() { return inputNamesCStr; }

std::vector<Ort::Value> OnnxModelBase::forward(std::vector<Ort::Value>& inputTensors)
{
  // todo: make runOptions parameter here

  return session.Run(Ort::RunOptions{nullptr},
                     inputNamesCStr.data(),
                     inputTensors.data(),
                     inputNamesCStr.size(),
                     outputNamesCStr.data(),
                     outputNamesCStr.size());
}

// OnnxModelBase::~OnnxModelBase() {
//    // empty body
//}

} // namespace yolov8_onnxruntime