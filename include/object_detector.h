#pragma once

#include <net.h>
#include <opencv2/opencv.hpp>
#include <vector>

enum class Model { XS, SM, MD, LG, XL };

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

struct ObjectDetector {
  void init();

  void detect(cv::Mat &image, std::vector<Object> &objects);

  bool useCuda = true;
  Model model = Model::SM;

private:
  ncnn::Net ncnnNet;
  cv::dnn::Net net;
  std::vector<std::string> coco_list;

  std::vector<std::string> load_class_list();
  void load_net();

  void detectViaNcnn(cv::Mat &image, std::vector<Object> &objects);
  void detectViaCuda(cv::Mat &image, std::vector<Object> &objects);

  cv::dnn::Net loadCudeModel();
  void loadNcnnModel();

  void generate_proposal(const ncnn::Mat &pred, int stride,
                         const ncnn::Mat &in_pad, float prob_threshold,
                         std::vector<Object> objects);
  void generate_proposal(const ncnn::Mat &pred, std::vector<int> &strides,
                         const ncnn::Mat &in_pad, float prob_threshold,
                         std::vector<Object> &objects);
};
