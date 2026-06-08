#include "imgui-SFML.h"
#include "imgui.h"

#include <SFML/Graphics.hpp>
#include <opencv2/opencv.hpp>

sf::Texture matToTexture(const cv::Mat &frame) {
  cv::Mat rgb;
  cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGBA);

  sf::Image image;
  sf::Texture texture(sf::Vector2u(rgb.cols, rgb.rows));
  texture.update(rgb.data);
  return texture;
}

int main() {
  sf::RenderWindow window(sf::VideoMode::getDesktopMode(),
                          "traffic Surveillance");

  window.setPosition({1920, 0});

  window.setFramerateLimit(60);

  ImGui::SFML::Init(window);
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  sf::Clock deltaClock;

  // capture video
  cv::VideoCapture cap(DATA_PATH "demo1.mp4");
  cv::VideoCapture cap2(DATA_PATH "demo2.mp4");

  cv::Mat frame;

  while (window.isOpen()) {

    cap >> frame;

    if (frame.empty()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }

    sf::Texture tex = matToTexture(frame);

    sf::Time dt = deltaClock.restart();

    while (const std::optional event = window.pollEvent()) {
      ImGui::SFML::ProcessEvent(window, *event);
      if (event->is<sf::Event::Closed>())
        window.close();
    }

    ImGui::SFML::Update(window, dt);

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    sf::Vector2u size = window.getSize();

    ImGui::SetNextWindowSize(ImVec2(size.x, size.y), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);

    if (ImGui::Begin("Main window", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar)) {

      ImGui::Image(tex, sf::Vector2f(300, 200));
    }

    ImGui::End();

    window.clear();

    ImGui::SFML::Render(window);
    window.display();
  }

  ImGui::SFML::Shutdown();

  return 0;
}
