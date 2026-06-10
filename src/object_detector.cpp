#include "object_detector.h"
#include <fstream>
#include <gpu.h>
#include <string.h>
#include <utility>

const int target_size = 640;
const float SCORE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.45f;

static const char *class_names[] = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush",
};

static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void ObjectDetector::init() { load_net(); }

void ObjectDetector::load_net() {

  if (useCuda) {
    std::cout << "Using Cuda" << std::endl;
    auto result = loadCudeModel();

    result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    net = result;

    coco_list = load_class_list();
  } else {
    std::cout << "Using vulkan" << std::endl;
    ncnnNet.opt.use_vulkan_compute = true;
    loadNcnnModel();
  }
}

cv::dnn::Net ObjectDetector::loadCudeModel() {
  switch (model) {
  case Model::XS:
    std::cout << "yolo11 extra small model" << std::endl;
    return cv::dnn::readNetFromONNX(DATA_PATH "yolo_onnx_models/yolo11n.onnx");
  case Model::SM:
    std::cout << "yolo11 small model" << std::endl;
    return cv::dnn::readNetFromONNX(DATA_PATH "yolo_onnx_models/yolo11s.onnx");
  case Model::MD:
    std::cout << "yolo11 medium model" << std::endl;
    return cv::dnn::readNetFromONNX(DATA_PATH "yolo_onnx_models/yolo11m.onnx");
  case Model::LG:
    std::cout << "yolo11 large model" << std::endl;
    return cv::dnn::readNetFromONNX(DATA_PATH "yolo_onnx_models/yolo11l.onnx");
  case Model::XL:
    std::cout << "yolo11 extra large model" << std::endl;
    return cv::dnn::readNetFromONNX(DATA_PATH "yolo_onnx_models/yolo11x.onnx");
  }

  assert(false);
}

void ObjectDetector::loadNcnnModel() {
  switch (model) {
  case Model::XS:
    std::cout << "yolo11 extra small model" << std::endl;
    ncnnNet.load_param(DATA_PATH
                       "yolo_ncnn_models/yolo11n_ncnn_model/model.ncnn.param");
    ncnnNet.load_model(DATA_PATH
                       "yolo_ncnn_models/yolo11n_ncnn_model/model.ncnn.bin");
    return;
  case Model::SM:
    std::cout << "yolo11 small model" << std::endl;
    ncnnNet.load_param(DATA_PATH
                       "yolo_ncnn_models/yolo11s_ncnn_model/model.ncnn.param");
    ncnnNet.load_model(DATA_PATH
                       "yolo_ncnn_models/yolo11s_ncnn_model/model.ncnn.bin");
    return;
  case Model::MD:
    std::cout << "yolo11 medium model" << std::endl;
    ncnnNet.load_param(DATA_PATH
                       "yolo_ncnn_models/yolo11m_ncnn_model/model.ncnn.param");
    ncnnNet.load_model(DATA_PATH
                       "yolo_ncnn_models/yolo11m_ncnn_model/model.ncnn.bin");
    return;
  case Model::LG:
    std::cout << "yolo11 large model" << std::endl;
    ncnnNet.load_param(DATA_PATH
                       "yolo_ncnn_models/yolo11l_ncnn_model/model.ncnn.param");
    ncnnNet.load_model(DATA_PATH
                       "yolo_ncnn_models/yolo11l_ncnn_model/model.ncnn.bin");
    return;
  case Model::XL:
    std::cout << "yolo11 extra large model" << std::endl;
    ncnnNet.load_param(DATA_PATH
                       "yolo_ncnn_models/yolo11x_ncnn_model/model.ncnn.param");
    ncnnNet.load_model(DATA_PATH
                       "yolo_ncnn_models/yolo11x_ncnn_model/model.ncnn.bin");
    return;
  }

  assert(false);
}

cv::Mat format_yolov(const cv::Mat &source) {
  int col = source.cols;
  int row = source.rows;

  int _max = MAX(col, row);

  cv::Mat result(_max, _max, CV_8UC3);
  source.copyTo(result(cv::Rect(0, 0, col, row)));
  return result;
}

void ObjectDetector::detect(cv::Mat &image, std::vector<Object> &objects) {
  if (useCuda)
    detectViaCuda(image, objects);
  else
    detectViaNcnn(image, objects);
}

static void nms_sorted_bboxes(const std::vector<Object> &objects,
                              std::vector<int> &picked, float nms_threshold) {
  picked.clear();
  const int n = objects.size();
  std::vector<float> areas(n);

  for (int i = 0; i < n; i++)
    areas[i] = objects[i].rect.width * objects[i].rect.height;

  for (int i = 0; i < n; i++) {
    const Object &a = objects[i];
    bool keep = true;

    for (int j = 0; j < (int)picked.size(); j++) {
      const Object &b = objects[picked[j]];

      // intersection
      float inter_x0 = std::max(a.rect.x, b.rect.x);
      float inter_y0 = std::max(a.rect.y, b.rect.y);
      float inter_x1 =
          std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
      float inter_y1 =
          std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);

      float inter_w = std::max(0.0f, inter_x1 - inter_x0);
      float inter_h = std::max(0.0f, inter_y1 - inter_y0);
      float inter_area = inter_w * inter_h;

      float iou = inter_area / (areas[i] + areas[picked[j]] - inter_area);

      if (iou > nms_threshold) {
        keep = false;
        break;
      }
    }

    if (keep)
      picked.push_back(i);
  }
}

void ObjectDetector::detectViaNcnn(cv::Mat &image,
                                   std::vector<Object> &objects) {

  int img_w = image.cols;
  int img_h = image.rows;

  int w = img_w, h = img_h;
  float scale = std::min((float)target_size / w, (float)target_size / h);
  w = w * scale;
  h = h * scale;

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      image.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

  // letter boxing padding to 640x640
  int wpad = target_size - w;
  int hpad = target_size - h;

  ncnn::Mat in_pad;
  ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

  const float norm_vals[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
  in_pad.substract_mean_normalize(0, norm_vals);

  // inference
  ncnn::Extractor ex = ncnnNet.create_extractor();
  ex.input("in0", in_pad);

  ncnn::Mat out;
  ex.extract("out0", out);

  std::vector<Object> proposals;
  const int num_class = 80;

  for (int i = 0; i < out.w; i++) {
    float cx = out.row(0)[i];
    float cy = out.row(1)[i];
    float bw = out.row(2)[i];
    float bh = out.row(3)[i];

    int label = -1;
    float score = -FLT_MAX;
    for (int k = 0; k < num_class; k++) {
      float s = out.row(4 + k)[i];
      if (s > score) {
        score = s;
        label = k;
      }
    }

    if (score < SCORE_THRESHOLD)
      continue;

    // scale box back to original image coordinates
    float x0 = (cx - bw * 0.5f - wpad / 2.0f) / scale;
    float y0 = (cy - bh * 0.5f - hpad / 2.0f) / scale;
    float x1 = (cx + bw * 0.5f - wpad / 2.0f) / scale;
    float y1 = (cy + bh * 0.5f - hpad / 2.0f) / scale;

    x0 = std::max(0.f, x0);
    y0 = std::max(0.f, y0);
    x1 = std::min((float)img_w, x1);
    y1 = std::min((float)img_h, y1);

    Object obj;
    obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
    obj.label = label;
    obj.prob = score;
    proposals.push_back(obj);
  }

  std::sort(proposals.begin(), proposals.end(),
            [](const Object &a, const Object &b) { return a.prob > b.prob; });

  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, NMS_THRESHOLD);

  objects.resize(picked.size());
  for (int i = 0; i < (int)picked.size(); i++)
    objects[i] = proposals[picked[i]];
}

void ObjectDetector::draw_object(cv::Mat &image, std::vector<Object> &objects) {
  static const std::vector<int> vehicle_classes = {2, 3, 5, 7};

  std::map<int, cv::Scalar> colors = {
      {2, cv::Scalar(0, 255, 0)},
      {3, cv::Scalar(255, 0, 0)},
      {5, cv::Scalar(0, 165, 255)},
      {7, cv::Scalar(0, 0, 255)},
  };

  for (const Object &obj : objects) {

    if (std::find(vehicle_classes.begin(), vehicle_classes.end(), obj.label) ==
        vehicle_classes.end())
      continue;

    cv::Scalar color = colors[obj.label];
    cv::rectangle(image, obj.rect, color, 2);

    char text[64];
    snprintf(text, sizeof(text), "%s %.1f%%", class_names[obj.label],
             obj.prob * 100.f);

    int baseline = 0;
    cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int x = obj.rect.x;
    int y = obj.rect.y - 5;
    y = std::max(y, text_size.height);

    cv::rectangle(image, cv::Point(x, y - text_size.height),
                  cv::Point(x + text_size.width, y + baseline), color, -1);
    cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
  }
}

void ObjectDetector::detectViaCuda(cv::Mat &image,
                                   std::vector<Object> &objects) {
  int img_w = image.cols;
  int img_h = image.rows;

  float scale =
      std::min((float)target_size / img_w, (float)target_size / img_h);
  int w = img_w * scale;
  int h = img_h * scale;
  int wpad = target_size - w;
  int hpad = target_size - h;

  cv::Mat resized, padded;
  cv::resize(image, resized, cv::Size(w, h));
  cv::copyMakeBorder(resized, padded, hpad / 2, hpad - hpad / 2, wpad / 2,
                     wpad - wpad / 2, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  cv::Mat blob;
  cv::dnn::blobFromImage(padded, blob, 1.0f / 255.0f,
                         cv::Size(target_size, target_size), cv::Scalar(), true,
                         false);
  net.setInput(blob);
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  cv::Mat pred = outputs[0].reshape(1, 84);
  cv::transpose(pred, pred);

  float *data = (float *)pred.data;
  std::vector<Object> proposals;

  for (int i = 0; i < 8400; i++) {
    float cx = data[0];
    float cy = data[1];
    float bw = data[2];
    float bh = data[3];

    float *class_scores = data + 4;
    cv::Mat scores(1, 80, CV_32FC1, class_scores);
    cv::Point class_id;
    double max_score;
    cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);

    if (max_score < SCORE_THRESHOLD) {
      data += 84;
      continue;
    }

    float x0 = (cx - bw * 0.5f - wpad / 2.0) / scale;
    float y0 = (cy - bh * 0.5f - hpad / 2.0) / scale;
    float x1 = (cx + bw * 0.5f - wpad / 2.0) / scale;
    float y1 = (cy + bh * 0.5f - hpad / 2.0) / scale;

    x0 = std::max(0.0f, x0);
    y0 = std::max(0.0f, y0);
    x1 = std::min((float)img_w, x1);
    y1 = std::min((float)img_h, y1);

    Object obj;
    obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
    obj.label = class_id.x;
    obj.prob = (float)max_score;
    proposals.push_back(obj);

    data += 84;
  }

  std::sort(proposals.begin(), proposals.end(),
            [](const Object &a, const Object &b) { return a.prob > b.prob; });

  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, NMS_THRESHOLD);

  objects.resize(picked.size());
  for (int i = 0; i < (int)picked.size(); i++)
    objects[i] = proposals[picked[i]];
}

std::vector<std::string> ObjectDetector::load_class_list() {
  std::vector<std::string> coco_list;
  std::ifstream file(DATA_PATH "coco.txt");
  if (!file) {
    std::cerr << "failed to open file" << std::endl;
    return {};
  }

  std::string line;
  while (std::getline(file, line))
    coco_list.push_back(line);

  return coco_list;
}
