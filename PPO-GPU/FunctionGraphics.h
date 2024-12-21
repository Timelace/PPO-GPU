#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics.hpp>
#include <functional>
#include <chrono>
#include <cstdarg>
#include <thread>
#include <atomic>

class FunctionGraphics {
public:
	FunctionGraphics(int width, int height, float x_min, float x_max, float y_min, float y_max, int points, std::function<float(float input)> equation);
	~FunctionGraphics();

	void set_external_points(float* y_values);

	float get_x_min();
	float get_x_max();
private:

	sf::RenderWindow window;
	sf::View graph_view;
	int m_height, m_width;
	float m_x_min, m_x_max, m_y_min, m_y_max;
	float m_x_dist, m_y_dist;
	int m_points;
	std::function<float(float input)> m_equation;
	sf::Font font;
	sf::Text timer_txt;
	float s_time;

	float* external_points;
	void draw_input();
	std::atomic<bool> has_data = false;

	bool do_update = false;

	std::thread main_loop_thd;

	void event_loop();
	void reset_graph();

	// event handlers
	void resize(int new_width, int new_height);
	void scroll(float amount);
	float scroll_factor = 0.9f;
	void move(int mouse_x, int mouse_y);
	int pressed_pos[2];
	bool pressed = false;
};