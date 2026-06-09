#include "object_detector.h"
#include <fstream>
#include <gpu.h>
#include <string.h>
#include <utility>

static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void ObjectDetector::init() {
  coco_list = load_class_list();
  load_net();
}

void ObjectDetector::load_net() {

  if (useCuda) {
    std::cout << "Using Cuda" << std::endl;
    auto result = loadCudeModel();

    result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    net = result;
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

void ObjectDetector::detectViaNcnn(cv::Mat &image,
                                   std::vector<Object> &objects) {
  const int target_size = 640;
  const float prob_threshold = 0.25f;
  const float nms_threshold = 0.45f;

  int img_w = image.cols;
  int img_h = image.rows;

  std::vector<int> strides(3);
  strides[0] = 8;
  strides[1] = 16;
  strides[2] = 32;
  const int max_stride = 32;

  int w = img_w;
  int h = img_h;
  float scale = 1.f;

  if (w > h) {
    scale = (float)target_size / w;
    w = target_size;
    h = h * scale;
  } else {
    scale = (float)target_size / h;
    h = target_size;
    w = w * scale;
  }

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      image.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

  int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
  int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

  const float norm_vals[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
  in_pad.substract_mean_normalize(0, norm_vals);

  ncnn::Extractor ex = ncnnNet.create_extractor();
  ex.input("in0", in_pad);

  ncnn::Mat out;
  ex.extract("out0", out);

  std::vector<Object> proposals;
  // generate_proposal(out, strides, in_pad, prob_threshold, proposals);
  // std::cout << proposals.size() << std::endl;
}

void ObjectDetector::detectViaCuda(cv::Mat &image,
                                   std::vector<Object> &objects) {
  cv::Mat input_image = format_yolov(image);
  cv::Mat blob;

  cv::Size modelShape(640, 640);

  cv::dnn::blobFromImage(input_image, blob, 1. / 255, modelShape, cv::Scalar(),
                         true, false);
  net.setInput(blob);
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  float x_factor = input_image.cols / (float)modelShape.width;
  float y_factor = input_image.rows / (float)modelShape.height;

  float *data = (float *)outputs[0].data;
  //
  // const int dimension = 85;
  // const int rows = 25200;
  //
  // std::vector<int> class_ids;
  // std::vector<float> confidences;
  // std::vector<cv::Rect> boxes;
  //
  // return;
  //
  // for (int i = 0; i < rows; i++) {
  //
  //   float confidence = data[4];
  //
  //   if (confidence >= SCORE_THRESHOLD) {
  //
  //     float *classes_scores = data + 5;
  //     cv::Mat scores(1, coco_list.size(), CV_32FC1, classes_scores);
  //
  //     cv::Point class_id;
  //     double max_class_score;
  //     cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
  //
  //     if (max_class_score > SCORE_THRESHOLD) {
  //       confidences.push_back(confidence);
  //       class_ids.push_back(class_id.x);
  //
  //       float x = data[0];
  //       float y = data[1];
  //       float w = data[2];
  //       float h = data[2];
  //
  //       int left = int((x - 0.5 * w) * x_factor);
  //       int right = int((y - 0.5 * h) * y_factor);
  //       int width = int(w * x_factor);
  //       int height = int(h * y_factor);
  //
  //       boxes.push_back(cv::Rect(left, right, width, height));
  //     }
  //   }
  //
  //   data += dimension;
  // }

  // result
  // std::vector<int> nms_result;
  // cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
  //                   nms_result);
  // std::cout << nms_result.size() << std::endl;
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

void ObjectDetector::generate_proposal(const ncnn::Mat &pred, int stride,
                                       const ncnn::Mat &in_pad,
                                       float prob_threshold,
                                       std::vector<Object> objects) {
  const int w = in_pad.w;
  const int h = in_pad.h;

  const int num_grid_x = w / stride;
  const int num_grid_y = h / stride;

  const int reg_max_1 = 16;
  const int num_class =
      pred.w - reg_max_1 * 4; // number of classes. 80 for COCO

  for (int y = 0; y < num_grid_y; y++) {
    for (int x = 0; x < num_grid_x; x++) {
      const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

      // find label with max score
      int label = -1;
      float score = -FLT_MAX;
      {
        const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);

        for (int k = 0; k < num_class; k++) {
          float s = pred_score[k];
          if (s > score) {
            label = k;
            score = s;
          }
        }

        score = sigmoid(score);
      }

      if (score >= prob_threshold) {
        ncnn::Mat pred_bbox =
            pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4);

        {
          ncnn::Layer *softmax = ncnn::create_layer("Softmax");

          ncnn::ParamDict pd;
          pd.set(0, 1); // axis
          pd.set(1, 1);
          softmax->load_param(pd);

          ncnn::Option opt;
          opt.num_threads = 1;
          opt.use_packing_layout = false;

          softmax->create_pipeline(opt);

          softmax->forward_inplace(pred_bbox, opt);

          softmax->destroy_pipeline(opt);

          delete softmax;
        }

        float pred_ltrb[4];
        for (int k = 0; k < 4; k++) {
          float dis = 0.f;
          const float *dis_after_sm = pred_bbox.row(k);
          for (int l = 0; l < reg_max_1; l++) {
            dis += l * dis_after_sm[l];
          }

          pred_ltrb[k] = dis * stride;
        }

        float pb_cx = (x + 0.5f) * stride;
        float pb_cy = (y + 0.5f) * stride;

        float x0 = pb_cx - pred_ltrb[0];
        float y0 = pb_cy - pred_ltrb[1];
        float x1 = pb_cx + pred_ltrb[2];
        float y1 = pb_cy + pred_ltrb[3];

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = label;
        obj.prob = score;

        objects.push_back(obj);
      }
    }
  }
}

void ObjectDetector::generate_proposal(const ncnn::Mat &pred,
                                       std::vector<int> &strides,
                                       const ncnn::Mat &in_pad,
                                       float prob_threshold,
                                       std::vector<Object> &objects) {
  const int w = in_pad.w;
  const int h = in_pad.h;

  int pred_row_offset = 0;
  for (size_t i = 0; i < strides.size(); i++) {
    const int stride = strides[i];

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;
    const int num_grid = num_grid_x * num_grid_y;

    generate_proposal(pred.row_range(pred_row_offset, num_grid), stride, in_pad,
                      prob_threshold, objects);
    pred_row_offset += num_grid;
  }
}
