#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics.hpp>
#include <thread>
#include <atomic>
#include <cstdarg>
#include <functional>

class CartPole {
public:
	CartPole(int width, int height);
	~CartPole();

	void join_window();

private:
	sf::RenderWindow window;
	sf::View game_view;
	int m_width, m_height;
	sf::Font timer_font;
	sf::Text timer_text;
	float time = 0;

	sf::RectangleShape cart[3];
	sf::RectangleShape pole;
	void init_shapes();

	void main_loop();
	std::thread main_loop_thd;

	void draw_frame();
	void calculate_physics();

	float max_velocity_per_second;

	float m_pole_pos_x;
	float m_pole_pos_y;
	float m_pole_width;
	float m_pole_height;

	float m_cart_pos_x;
	float m_cart_pos_y;
	float m_cart_width;
	float m_cart_height;
	float m_cart_wall_thickness;

	float m_cart_mass;
	float m_cart_friction;
	float m_cart_velocity;
	float m_cart_acceleration;
	float m_cart_normal;
	float m_cart_velocity_scale = 10.0f;
	float m_cart_position_scale = 10.0f;

	float m_pole_mass;
	float m_pole_length;
	float m_pole_friction;
	float m_angle;
	float m_angular_velocity;
	float m_angular_acceleration;
	float m_angular_velocity_scale = 0.1f;
	float m_angular_position_scale = 0.01f;

	float m_gravity;
	float m_force;

	void key_pressed_handler(sf::Event key_event);
	void key_released_handler(sf::Event key_event);
};
