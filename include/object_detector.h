#pragma once

#include <net.h>
#include <opencv2/opencv.hpp>
#include <vector>

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

struct ObjectDetector {
  void init();

  void detect(cv::Mat &image, std::vector<Object> &objects);

  bool useCuda = false;

private:
  ncnn::Net ncnnNet;
  cv::dnn::Net net;
  std::vector<std::string> coco_list;

  std::vector<std::string> load_class_list();
  void load_net();

  void detectViaNcnn(cv::Mat &image, std::vector<Object> &objects);
  void detectViaCuda(cv::Mat &image, std::vector<Object> &objects);
};
