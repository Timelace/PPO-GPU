#include "FunctionGraphics.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <SFML/Window/Mouse.hpp>
#include <string>

FunctionGraphics::FunctionGraphics(int width, int height, float x_min, float x_max, float y_min, float y_max, int points, std::function<float(float input)> equation) :
	m_height(height),
	m_width(width),
	m_x_min(x_min),
	m_x_max(x_max),
	m_y_min(y_min),
	m_y_max(y_max),
	m_points(points),
	m_equation(equation)
{

	m_x_dist = std::abs(x_min - x_max);
	m_y_dist = std::abs(y_min - y_max);

	font.loadFromFile("E:/Machine_Learning_Cpp/temporary/PPO-GPU/x64/Fonts/7segment.ttf");
	//if (!font.loadFromFile("../Fonts/7segment.ttf"))
		//font.loadFromFile("../x64/Fonts/7segment.ttf");
	timer_txt.setFont(font);
	timer_txt.setCharacterSize((height > width) ? width * 0.05 : height * 0.05);
	timer_txt.setColor(sf::Color::White);
	timer_txt.setPosition(width * 0.02, height * 0.02);
	timer_txt.setString("0.00");

	s_time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;

	external_points = (float*)malloc(points * sizeof(float));

	main_loop_thd = std::thread(&FunctionGraphics::event_loop, this);
}

FunctionGraphics::~FunctionGraphics() {
	window.close();
	free(external_points);
}

void FunctionGraphics::event_loop() {
	window.create(sf::VideoMode(m_width, m_height), "Function Graph");
	window.setVerticalSyncEnabled(true);
	graph_view.reset(sf::FloatRect(0, 0, m_width, m_height));
	window.setView(graph_view);
	reset_graph();
	window.display();
	sf::Event events;
	while (window.isOpen()) {
		while (window.pollEvent(events)) {
			if (events.type == sf::Event::Closed)
				window.close();
			if (events.type == sf::Event::Resized)
				resize(events.size.width, events.size.height);
			if (events.type == sf::Event::MouseWheelScrolled)
				scroll(events.mouseWheelScroll.delta);
			if (events.type == sf::Event::MouseButtonPressed) {
				pressed_pos[0] = sf::Mouse::getPosition().x;
				pressed_pos[1] = sf::Mouse::getPosition().y;
				pressed = true;
			}
			if (events.type == sf::Event::MouseButtonReleased)
				pressed = false;
			if (events.type == sf::Event::MouseMoved)
				move(sf::Mouse::getPosition().x, sf::Mouse::getPosition().y);
		}
		reset_graph();
		//if (has_data)
			draw_input();
		window.display();
	}
}

void FunctionGraphics::set_external_points(float* y_values) {
	if (!has_data) {
		std::memcpy(external_points, y_values, m_points * sizeof(float));
		has_data = true;
	}
}

void FunctionGraphics::draw_input() {
	sf::Vertex* eq_lines = (sf::Vertex*)malloc(m_points * sizeof(sf::Vertex));
	for (int p = 0; p < m_points; p++) {
		float x_coord = ((float)m_width / m_points) * p;
		float y_coord = ((float)m_height / m_y_dist) * (m_y_max - external_points[p]);
		eq_lines[p].position = sf::Vector2f(x_coord, y_coord);
		eq_lines[p].color = sf::Color(44, 171, 93);
	}
	window.draw(eq_lines, m_points, sf::LineStrip);
	has_data = false;
}

void FunctionGraphics::reset_graph() {
	window.clear(sf::Color::Black);
	int bg_lines_size = (int)m_x_dist + (int)m_y_dist + 2;
	sf::Vertex* bg_lines = (sf::Vertex*)malloc(bg_lines_size * sizeof(sf::Vertex) * 2);
	for (int x = 0; x <= (int)m_x_dist; x++) {
		float x_coord = (m_width / m_x_dist) * (x + ((m_x_min < 0) ? std::abs(m_x_min - (int)m_x_min) : 1 - std::abs(m_x_min - (int)m_x_min)));
		bg_lines[x * 2].position = sf::Vector2f(x_coord, m_height);
		bg_lines[x * 2].color = (m_x_min + (x + std::abs(m_x_min - (int)m_x_min)) == 0.0) ? sf::Color(255, 255, 255) : sf::Color(51, 51, 50);
		bg_lines[x * 2 + 1].position = sf::Vector2f(x_coord, 0);
		bg_lines[x * 2 + 1].color = (m_x_min + (x + std::abs(m_x_min - (int)m_x_min)) == 0.0) ? sf::Color(255, 255, 255) : sf::Color(51, 51, 50);
	}
	for (int y = 0; y <= (int)m_y_dist; y++) {
		float y_coord = (m_height / m_y_dist) * (y + ((m_y_max > 0) ? std::abs(m_y_max - (int)m_y_max) : 1 - std::abs(m_y_max - (int)m_y_max)));
		bg_lines[(y + (int)m_x_dist) * 2 + 2].position = sf::Vector2f(m_width, y_coord);
		bg_lines[(y + (int)m_x_dist) * 2 + 2].color = (m_y_max - (y + std::abs(m_y_max - (int)m_y_max)) == 0.0) ? sf::Color(255, 255, 255) : sf::Color(51, 51, 50);
		bg_lines[(y + (int)m_x_dist) * 2 + 3].position = sf::Vector2f(0, y_coord);
		bg_lines[(y + (int)m_x_dist) * 2 + 3].color = (m_y_max - (y + std::abs(m_y_max - (int)m_y_max)) == 0.0) ? sf::Color(255, 255, 255) : sf::Color(51, 51, 50);
	}
	window.draw(bg_lines, bg_lines_size * 2, sf::Lines);
	sf::Vertex* eq_lines = (sf::Vertex*)malloc(m_points * sizeof(sf::Vertex));
	for (int p = 0; p < m_points; p++) {
		float x_coord = ((float)m_width / m_points) * p;
		float y_coord = ((float)m_height / m_y_dist) * (m_y_max - m_equation((float)m_x_dist / m_width * x_coord + m_x_min));
		eq_lines[p].position = sf::Vector2f(x_coord, y_coord);
		eq_lines[p].color = sf::Color(4, 206, 209);
	}
	window.draw(eq_lines, m_points, sf::LineStrip);
	float c_time = ((std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f) - s_time;
	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << c_time;
	timer_txt.setString(ss.str());
	window.draw(timer_txt);
	free(bg_lines);
	free(eq_lines);
}

void FunctionGraphics::resize(int new_width, int new_height) {
	int window_center_x = window.getPosition().x + (m_width / 2);
	int window_center_y = window.getPosition().y + (m_height / 2);
	int mouse_x = sf::Mouse::getPosition().x;
	int mouse_y = sf::Mouse::getPosition().y;
	if (mouse_x > window_center_x) {
		m_x_max += (m_x_dist / m_width) * (new_width - m_width);
	}else {
		m_x_min -= (m_x_dist / m_width) * (new_width - m_width);
	}
	m_x_dist += (m_x_dist / m_width) * (new_width - m_width);
	m_width = new_width;
	if (mouse_y < window_center_y) {
		m_y_max += (m_y_dist / m_height) * (new_height - m_height);
	}
	else {
		m_y_min -= (m_y_dist / m_height) * (new_height - m_height);
	}
	m_y_dist += (m_y_dist / m_height) * (new_height - m_height);
	m_height = new_height;
	timer_txt.setCharacterSize((m_height > m_width) ? m_width * 0.05 : m_height * 0.05);
	timer_txt.setPosition(m_width * 0.02, m_height * 0.02);
	graph_view.reset(sf::FloatRect(0, 0, m_width, m_height));
	window.setView(graph_view);
}

void FunctionGraphics::scroll(float amount) {
	float scale = (amount < 0) ? 2.0f - scroll_factor : scroll_factor;
	m_x_min *= scale;
	m_x_max *= scale;
	m_x_dist *= scale;
	m_y_min *= scale;
	m_y_max *= scale;
	m_y_dist *= scale;
}

void FunctionGraphics::move(int mouse_x, int mouse_y) {
	if (!pressed) {
		return;
	}
	int delta_x = mouse_x - pressed_pos[0];
	int delta_y = mouse_y - pressed_pos[1];
	m_x_min -= (m_x_dist / m_width) * delta_x;
	m_x_max -= (m_x_dist / m_width) * delta_x;
	m_y_min += (m_y_dist / m_height) * delta_y;
	m_y_max += (m_y_dist / m_height) * delta_y;
	pressed_pos[0] = mouse_x;
	pressed_pos[1] = mouse_y;
}

float FunctionGraphics::get_x_min() {
	return m_x_min;
}

float FunctionGraphics::get_x_max() {
	return m_x_max;
}