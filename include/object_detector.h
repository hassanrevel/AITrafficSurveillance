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
  void draw_object(cv::Mat &image, std::vector<Object> &objects);

  bool useCuda = false;
  uint32_t yoloVersion = 11;
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
};
