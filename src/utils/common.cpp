#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <string>
#include <sstream>
#include <codecvt>
#include "utils/common.h"

#include <chrono>
//#include <boost/algorithm/string.hpp>

#include <iostream>
#include <stdio.h>

#include <string>

#include <regex>
#include <vector>



Timer::Timer(double& accumulator, bool isEnabled)
    : accumulator(accumulator), isEnabled(isEnabled) {
    if (isEnabled) {
        start = std::chrono::high_resolution_clock::now();
    }
}

// Stop the timer and update the accumulator
void Timer::Stop() {
    if (isEnabled) {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        accumulator += duration;
    }
}

// С++ 14 version
//#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
std::wstring get_win_path(const std::string& modelPath) {
#ifdef _WIN32
    return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(modelPath);
#else
    // return modelPath;
    return std::wstring(modelPath.begin(), modelPath.end());
#endif
}


std::vector<std::string> parseVectorString(const std::string& input) {
    /* Main purpose of this function is to parse `imgsz` key value of model metadata
     *  and from [height, width] get height, width values in the vector of strings
     * Args:
     *  input:
     *      expected to be something like [544, 960] or [3,544, 960]
     * output:
     *  iterable of strings, representing integers
     */
    std::regex number_pattern(R"(\d+)");

    std::vector<std::string> result;
    std::sregex_iterator it(input.begin(), input.end(), number_pattern);
    std::sregex_iterator end;

    while (it != end) {
        result.push_back(it->str());
        ++it;
    }

    return result;
}

std::vector<int> convertStringVectorToInts(const std::vector<std::string>& input) {
    std::vector<int> result;

    for (const std::string& str : input) {
        try {
            int value = std::stoi(str);
            result.push_back(value);
        }
        catch (const std::invalid_argument& e) {
            // raise explicit exception
            throw std::invalid_argument("Bad argument (cannot cast): value=" + str);
        }
        catch (const std::out_of_range& e) {
            // check bounds
            throw std::out_of_range("Value out of range: " + str);
        }
    }

    return result;
}


/*
std::unordered_map<int, std::string> parseNames(const std::string& input) {
    std::unordered_map<int, std::string> result;

    std::string cleanedInput = input;
    boost::erase_all(cleanedInput, "{");
    boost::erase_all(cleanedInput, "}");

    std::vector<std::string> elements;
    boost::split(elements, cleanedInput, boost::is_any_of(","));

    for (const std::string& element : elements) {
        std::vector<std::string> keyValue;
        boost::split(keyValue, element, boost::is_any_of(":"));

        if (keyValue.size() == 2) {
            int key = std::stoi(boost::trim_copy(keyValue[0]));
            std::string value = boost::trim_copy(keyValue[1]);

            result[key] = value;
        }
    }

    return result;
}
*/

std::unordered_map<int, std::string> parseNames(const std::string& input) {
    std::unordered_map<int, std::string> result;

    std::string cleanedInput = input;
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '{'), cleanedInput.end());
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '}'), cleanedInput.end());

    std::istringstream elementStream(cleanedInput);
    std::string element;
    while (std::getline(elementStream, element, ',')) {
        std::istringstream keyValueStream(element);
        std::string keyStr, value;
        if (std::getline(keyValueStream, keyStr, ':') && std::getline(keyValueStream, value)) {
            int key = std::stoi(keyStr);
            result[key] = value;
        }
    }

    return result;
}

int64_t vector_product(const std::vector<int64_t>& vec) {
    int64_t result = 1;
    for (int64_t value : vec) {
        result *= value;
    }
    return result;
}
