// =============================================================================
// 🐍 Snake Detector — YOLOv8 NCNN Header
// MODE: OBJECT DETECTION (bounding boxes + labels)
// Output: Draws boxes around snakes like your reference screenshot
// =============================================================================

#ifndef SNAKE_DETECTOR_H
#define SNAKE_DETECTOR_H

#include <opencv2/core/core.hpp>
#include <net.h>
#include <string>
#include <vector>

// Detection result — one per detected snake
struct Object
{
    cv::Rect_<float> rect;  // Bounding box (x, y, width, height)
    int label;              // Class index (0=venomous, 1=non_venomous)
    float prob;             // Confidence 0.0-1.0
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

// YOLOv8 NCNN Detector
class SnakeDetector
{
public:
    SnakeDetector();
    ~SnakeDetector();

    int load(const std::string& param_path, const std::string& bin_path,
             int target_size = 320, int num_threads = 4);

    int detect(const cv::Mat& bgr, std::vector<Object>& objects,
               float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int draw(cv::Mat& bgr, const std::vector<Object>& objects);

    static const char* getClassName(int label);
    static bool isVenomous(int label);
    
    // Number of classes — set after loading model
    int num_classes;

private:
    ncnn::Net net;
    int target_size;
    float norm_vals[3];
};

#endif // SNAKE_DETECTOR_H
