#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * This interface must provide only required arguments to load any onnx model regarding specific info -
 *  - i.e. modelPath will always be required, provider like "cpu" or "cuda" the same, since these are parameters you need
 *  to set up `sessionOptions` or `session` objects properly, but image size is not needed for pure onnx graph to be loaded so do NOT include it here
 */
class OnnxModelBase {
public:
    OnnxModelBase(const char* modelPath, const char* logid, const char* provider);
    //OnnxModelBase();  // no default constructor should be there
    //virtual ~OnnxModelBase();
    virtual const std::vector<std::string>& getInputNames(); // = 0
    virtual const std::vector<std::string>& getOutputNames();
    virtual const std::vector<const char*> getOutputNamesCStr();
    virtual const std::vector<const char*> getInputNamesCStr();
    virtual const Ort::ModelMetadata& getModelMetadata();
    virtual const std::unordered_map<std::string, std::string>& getMetadata();
    virtual const char* getModelPath();
    virtual const Ort::Session& getSession();
    //virtual std::vector<Ort::Value> forward(std::vector<Ort::Value> inputTensors);
    virtual std::vector<Ort::Value> forward(std::vector<Ort::Value>& inputTensors);
    Ort::Session session{ nullptr };

protected:
    const char* modelPath_;
    Ort::Env env{ nullptr };

    std::vector<std::string> inputNodeNames;
    std::vector<std::string> outputNodeNames;
    Ort::ModelMetadata model_metadata{ nullptr };
    std::unordered_map<std::string, std::string> metadata;
    std::vector<const char*> outputNamesCStr;
    std::vector<const char*> inputNamesCStr;
};
