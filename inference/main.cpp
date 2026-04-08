// =============================================================================
// Snake Detector — Main (Camera + Bounding Boxes + FPS)
// Draws boxes around snakes exactly like the reference screenshot
// =============================================================================

#include "snake_detector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <numeric>
#include <deque>
#include <signal.h>

static volatile bool g_running = true;
void sig_handler(int) { g_running = false; }

static void usage(const char* p) {
    fprintf(stderr, "Snake Detector — YOLOv8n + NCNN\n");
    fprintf(stderr, "Usage: %s [options]\n", p);
    fprintf(stderr, "  --model-dir DIR  Model dir (default: ./models)\n");
    fprintf(stderr, "  --camera ID      Camera (default: 0)\n");
    fprintf(stderr, "  --size N         Input size (default: 320)\n");
    fprintf(stderr, "  --conf F         Confidence (default: 0.4)\n");
    fprintf(stderr, "  --threads N      CPU threads (default: 4)\n");
    fprintf(stderr, "  --headless       No display\n");
    fprintf(stderr, "  --image PATH     Single image\n");
}

int main(int argc, char** argv)
{
    const char* model_dir = "./models";
    int camera_id = 0, target_size = 320, threads = 4;
    float conf = 0.4f;
    bool headless = false;
    const char* image_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model-dir") && i+1<argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--camera") && i+1<argc) camera_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i+1<argc) target_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--conf") && i+1<argc) conf = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1<argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--headless")) headless = true;
        else if (!strcmp(argv[i], "--image") && i+1<argc) image_path = argv[++i];
        else if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    fprintf(stderr, "\n=== Snake Detector — BIO_MIMIC ===\n\n");

    // Find model files
    char param[512], bin[512];
    const char* names[][2] = {
        {"model.ncnn.param","model.ncnn.bin"},
        {"best.ncnn.param","best.ncnn.bin"},
        {"yolov8n.param","yolov8n.bin"},
    };
    bool found = false;
    for (auto& n : names) {
        snprintf(param, 512, "%s/%s", model_dir, n[0]);
        snprintf(bin, 512, "%s/%s", model_dir, n[1]);
        FILE* f = fopen(param, "r");
        if (f) { fclose(f); found = true; break; }
    }
    if (!found) {
        fprintf(stderr, "No model in %s. Need .param + .bin files.\n", model_dir);
        return 1;
    }

    SnakeDetector det;
    if (det.load(param, bin, target_size, threads) != 0) return 1;

    // Single image mode
    if (image_path) {
        cv::Mat img = cv::imread(image_path, 1);
        if (img.empty()) { fprintf(stderr, "Cannot read %s\n", image_path); return 1; }

        std::vector<Object> objs;
        auto t0 = std::chrono::high_resolution_clock::now();
        det.detect(img, objs, conf, 0.5f);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float,std::milli>(t1-t0).count();

        det.draw(img, objs);
        fprintf(stderr, "Found %zu objects in %.1fms\n", objs.size(), ms);
        for (auto& o : objs)
            fprintf(stderr, "  %s %.0f%% [%.0f,%.0f %.0fx%.0f]\n",
                SnakeDetector::getClassName(o.label), o.prob*100,
                o.rect.x, o.rect.y, o.rect.width, o.rect.height);

        std::string out = std::string(image_path) + "_result.jpg";
        cv::imwrite(out, img);
        fprintf(stderr, "Saved: %s\n", out.c_str());
        return 0;
    }

    // Camera mode
    cv::VideoCapture cap;
    cap.open(camera_id, cv::CAP_V4L2);
    if (!cap.isOpened()) cap.open(camera_id);
    if (!cap.isOpened()) { fprintf(stderr, "Cannot open camera %d\n", camera_id); return 1; }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FPS, 30);
    fprintf(stderr, "Camera %d: %dx%d\n", camera_id,
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::deque<float> fps_hist;
    int fc = 0;

    while (g_running) {
        cv::Mat frame;
        cap.grab();
        if (!cap.retrieve(frame) || frame.empty()) continue;

        std::vector<Object> objs;
        auto t0 = std::chrono::high_resolution_clock::now();
        det.detect(frame, objs, conf, 0.5f);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float,std::milli>(t1-t0).count();
        float fps = 1000.f / ms;

        fps_hist.push_back(fps);
        if (fps_hist.size() > 30) fps_hist.pop_front();
        float avg = std::accumulate(fps_hist.begin(), fps_hist.end(), 0.f) / fps_hist.size();

        fc++;
        bool danger = false;
        for (auto& o : objs) if (SnakeDetector::isVenomous(o.label)) danger = true;

        fprintf(stderr, "\rFrame %d | %.1f FPS (avg %.1f) | %zu det | %s  ",
            fc, fps, avg, objs.size(), danger ? "VENOMOUS!" : "safe");
        fflush(stderr);

        if (!headless) {
            det.draw(frame, objs);

            char txt[64];
            sprintf(txt, "FPS: %.1f", avg);
            cv::putText(frame, txt, cv::Point(10,25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);

            if (danger) {
                cv::rectangle(frame, cv::Point(0, frame.rows-35),
                    cv::Point(frame.cols, frame.rows), cv::Scalar(0,0,200), -1);
                cv::putText(frame, "!! VENOMOUS SNAKE !!", cv::Point(15, frame.rows-10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);
            }

            cv::imshow("Snake Detector", frame);
            int k = cv::waitKey(1);
            if (k == 'q' || k == 27) break;
        }
    }

    fprintf(stderr, "\n\nDone. %d frames, avg %.1f FPS\n", fc,
        fps_hist.empty() ? 0.f :
        std::accumulate(fps_hist.begin(), fps_hist.end(), 0.f) / fps_hist.size());

    cap.release();
    if (!headless) cv::destroyAllWindows();
    return 0;
}
