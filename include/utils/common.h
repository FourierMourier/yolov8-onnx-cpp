#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>


class Timer {
public:
    Timer(double& accumulator, bool isEnabled = true);
    void Stop();

private:
    double& accumulator;
    bool isEnabled;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

std::wstring get_win_path(const std::string& path);
std::vector<std::string> parseVectorString(const std::string& input);
std::vector<int> convertStringVectorToInts(const std::vector<std::string>& input);
std::unordered_map<int, std::string> parseNames(const std::string& input);
int64_t vector_product(const std::vector<int64_t>& vec);
#endif // COMMON_H COMMON_UTILS_H