// =============================================================================
// Snake Detector — YOLOv8 NCNN Detection Engine
// For Ultralytics NCNN export (post-processed output)
// Output: out0 = [N x (4 + num_classes)] with decoded boxes + sigmoid scores
// =============================================================================

#include "snake_detector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

static const char* class_names[] = { "venomous", "non_venomous" };
static const cv::Scalar class_colors[] = {
    cv::Scalar(0, 0, 255),   // RED for venomous
    cv::Scalar(0, 255, 0)    // GREEN for non_venomous
};

static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left, j = right;
    float p = objects[(left + right) / 2].prob;
    while (i <= j) {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;
        if (i <= j) { std::swap(objects[i], objects[j]); i++; j--; }
    }
    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects,
    std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = objects[i].rect.width * objects[i].rect.height;

    for (int i = 0; i < n; i++) {
        const Object& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = objects[picked[j]];
            float inter = intersection_area(a, b);
            float u = areas[i] + areas[picked[j]] - inter;
            if (inter / u > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

SnakeDetector::SnakeDetector() : target_size(320), num_classes(2) {}
SnakeDetector::~SnakeDetector() { net.clear(); }

int SnakeDetector::load(const std::string& param_path, const std::string& bin_path,
                        int _target_size, int num_threads)
{
    net.clear();
    net.opt = ncnn::Option();
    net.opt.num_threads = num_threads;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = true;

    if (net.load_param(param_path.c_str()) != 0) {
        fprintf(stderr, "Failed to load param: %s\n", param_path.c_str());
        return -1;
    }
    if (net.load_model(bin_path.c_str()) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", bin_path.c_str());
        return -2;
    }
    target_size = _target_size;
    norm_vals[0] = 1.0f / 255.0f;
    norm_vals[1] = 1.0f / 255.0f;
    norm_vals[2] = 1.0f / 255.0f;
    fprintf(stderr, "Model loaded: %s (%dx%d, %d threads)\n",
            param_path.c_str(), target_size, target_size, num_threads);
    return 0;
}

int SnakeDetector::detect(const cv::Mat& rgb, std::vector<Object>& objects,
                          float prob_threshold, float nms_threshold)
{
    int width = rgb.cols, height = rgb.rows;
    int w = width, h = height;
    float scale = 1.f;
    if (w > h) { scale = (float)target_size / w; w = target_size; h = h * scale; }
    else       { scale = (float)target_size / h; h = target_size; w = w * scale; }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                  width, height, w, h);
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                           wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in_pad);
    ncnn::Mat out;
    ex.extract("out0", out);

    // Ultralytics NCNN export: out0 shape = [N x (4 + num_classes)]
    // Each row: [cx, cy, w, h, score_class0, score_class1, ...]
    // Boxes are already decoded (absolute coords in padded input space)
    // Scores are already sigmoid'd
    const int num_proposals = out.h;
    const int out_cols = out.w;  // should be 4 + num_classes = 6

    fprintf(stderr, "Output shape: %d x %d (expect Nx%d)\n", num_proposals, out_cols, 4 + num_classes);

    std::vector<Object> proposals;
    for (int i = 0; i < num_proposals; i++) {
        const float* row = out.row(i);

        // Find best class
        int label = -1;
        float best_score = -1.f;
        for (int k = 0; k < num_classes; k++) {
            float s = row[4 + k];  // scores start at index 4
            if (s > best_score) { label = k; best_score = s; }
        }

        if (best_score >= prob_threshold) {
            float cx = row[0];
            float cy = row[1];
            float bw = row[2];
            float bh = row[3];

            Object obj;
            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width = bw;
            obj.rect.height = bh;
            obj.label = label;
            obj.prob = best_score;
            proposals.push_back(obj);
        }
    }

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // Scale back to original image coordinates
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
        objects[i].rect.x = x0; objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0; objects[i].rect.height = y1 - y0;
    }
    return 0;
}

int SnakeDetector::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];
        int ci = obj.label % 2;
        const cv::Scalar& color = class_colors[ci];
        cv::rectangle(rgb, obj.rect, color, 3);

        char text[256];
        sprintf(text, "%s %.0f%%", class_names[ci], obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
        int x = obj.rect.x, y = obj.rect.y - label_size.height - baseLine - 4;
        if (y < 0) y = 0;
        if (x + label_size.width > rgb.cols) x = rgb.cols - label_size.width;
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
            cv::Size(label_size.width, label_size.height + baseLine + 4)), color, -1);
        cv::putText(rgb, text, cv::Point(x, y + label_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    return 0;
}

const char* SnakeDetector::getClassName(int label) {
    if (label >= 0 && label < 2) return class_names[label];
    return "unknown";
}

bool SnakeDetector::isVenomous(int label) { return label == 0; }
