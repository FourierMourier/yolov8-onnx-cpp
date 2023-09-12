#pragma once
#include <string>

namespace MetadataConstants {
    inline const std::string IMGSZ = "imgsz";
    inline const std::string STRIDE = "stride";
    inline const std::string NC = "nc";
    inline const std::string CH = "ch";
    inline const std::string DATE = "date";
    inline const std::string VERSION = "version";
    inline const std::string TASK = "task";
    inline const std::string BATCH = "batch";
    inline const std::string NAMES = "names";
}

namespace OnnxProviders {
    inline const std::string CPU = "cpu";
    inline const std::string CUDA = "cuda";
}

namespace OnnxInitializers
{
    inline const int UNINITIALIZED_STRIDE = -1;
    inline const int UNINITIALIZED_NC = -1;
}


namespace YoloTasks
{
    inline const std::string SEGMENT = "segment";
    inline const std::string DETECT = "detect";
    inline const std::string POSE = "pose";
    inline const std::string CLASSIFY = "classify";
}
