#include "CartPole.h"
#include <chrono>
#include <cmath>

CartPole::CartPole(int width, int height) :
	m_width(width),
	m_height(height)
{
	timer_font.loadFromFile("E:/Machine_Learning_Cpp/temporary/PPO-GPU/x64/Fonts/7segment.ttf");

	timer_text.setFont(timer_font);
	timer_text.setCharacterSize((height > width) ? width * 0.05 : height * 0.05);
	timer_text.setColor(sf::Color::White);
	timer_text.setPosition(width * 0.02, height * 0.02);
	timer_text.setString("0.00");

	time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;

	max_velocity_per_second = width * 0.004;

	m_cart_pos_x = width / 2.0;
	m_cart_pos_y = height - (height * 0.1);
	m_cart_width = width * 0.5;
	m_cart_height = height * 0.02;
	m_cart_wall_thickness = height * 0.02;

	m_pole_height = height - (height * 0.7);
	m_pole_width = width * 0.03;
	m_pole_pos_x = width / 2.0;
	m_pole_pos_y = m_cart_pos_y - m_cart_height - m_pole_height;

	m_cart_mass = 200.0f;
	m_cart_friction = 1.0f;
	m_cart_velocity = 0.0f;
	m_cart_acceleration = 0.0f;
	m_cart_normal = 0.0f;

	m_pole_mass = 20.0f;
	m_pole_length = 10.0f;
	m_pole_friction = 1.0f;
	m_angle = 0.0f;
	m_angular_velocity = 0.0f;
	m_angular_acceleration = 0.0f;

	m_gravity = 9.8f;
	m_force = 0.0f;

	main_loop_thd = std::thread(&CartPole::main_loop, this);
}

CartPole::~CartPole() {
	window.close();
}

void CartPole::main_loop() {
	window.create(sf::VideoMode(m_width, m_height), "Cart Pole");
	window.setVerticalSyncEnabled(true);
	game_view.reset(sf::FloatRect(0, 0, m_width, m_height));
	window.setView(game_view);
	init_shapes();
	window.display();
	sf::Event events;
	while (window.isOpen()) {
		while (window.pollEvent(events)) {
			if (events.type == sf::Event::Closed)
				window.close();
			if (events.type == sf::Event::KeyPressed)
				key_pressed_handler(events);
			if (events.type == sf::Event::KeyReleased)
				key_released_handler(events);
		}
		calculate_physics();
		draw_frame();
		window.display();
	}
}

void CartPole::init_shapes() {
	cart[0].setSize(sf::Vector2f(m_cart_wall_thickness, m_cart_height));
	cart[0].setOrigin(sf::Vector2f(-m_cart_pos_x, -m_cart_pos_y));
	cart[0].setPosition(sf::Vector2f(-m_cart_width / 2.0f, -m_cart_height * 2.0f));
	cart[0].setFillColor(sf::Color(43, 43, 43));
	cart[1].setSize(sf::Vector2f(m_cart_width, m_cart_height));
	cart[1].setOrigin(sf::Vector2f(-m_cart_pos_x, -m_cart_pos_y));
	cart[1].setPosition(sf::Vector2f(-m_cart_width / 2.0f, -m_cart_height));
	cart[1].setFillColor(sf::Color(43, 43, 43));
	cart[2].setSize(sf::Vector2f(m_cart_wall_thickness, m_cart_height));
	cart[2].setOrigin(sf::Vector2f(-m_cart_pos_x, -m_cart_pos_y));
	cart[2].setPosition(sf::Vector2f(m_cart_width / 2.0f - m_cart_wall_thickness, -m_cart_height * 2.0f));
	cart[2].setFillColor(sf::Color(43, 43, 43));
	window.draw(cart[0]);
	window.draw(cart[1]);
	window.draw(cart[2]);
	pole.setSize(sf::Vector2f(m_pole_width, m_pole_height));
	pole.setOrigin(sf::Vector2f(m_pole_width / 2.0f, m_pole_height / 2.0f));
	pole.setPosition(sf::Vector2f(m_cart_pos_x, m_cart_pos_y - m_cart_height - m_pole_height / 2.0f));
	window.draw(pole);
}

void CartPole::draw_frame() {
	window.clear(sf::Color::Black);
	cart[0].setOrigin(sf::Vector2f(-m_cart_pos_x, -m_cart_pos_y));
	cart[1].setOrigin(sf::Vector2f(-m_cart_pos_x, -m_cart_pos_y));
	cart[2].setOrigin(sf::Vector2f(-m_cart_pos_x, -m_cart_pos_y));
	window.draw(cart[0]);
	window.draw(cart[1]);
	window.draw(cart[2]);
	//sf::RectangleShape cart(sf::Vector2f(m_cart_width, m_cart_height));
	//cart.setFillColor(sf::Color(43, 43, 43));
	//cart.setPosition(sf::Vector2f(m_cart_pos_x - (m_cart_width / 2.0f), m_cart_pos_y - (m_cart_height / 2.0f)));
	//sf::RectangleShape pole(sf::Vector2f(m_pole_thickness, 250));

	//float angle = std::atan2(m_pole_pos_x - m_cart_pos_x, m_pole_pos_y - m_cart_pos_y);
	//float distance = std::sqrt((m_cart_pos_x - m_pole_pos_x) * (m_cart_pos_x - m_pole_pos_x) + ((m_cart_pos_y - m_cart_height) - m_pole_pos_y) * ((m_cart_pos_y - m_cart_height) - m_pole_pos_y));
	//float d1 = m_pole_width / 2.0f;
	//float l2 = distance - m_pole_height;
	//float y_offset = std::sqrt(d1 * d1 + l2 * l2) * std::sinf(angle - (3.14159263538979323f) + std::atanf(l2 / d1));

	//pole.setOrigin(m_pole_thickness / 2.0f, m_pole_height);
	//pole.setOrigin(m_pole_thickness / 2.0f, 250);
	//pole.setPosition(sf::Vector2f(m_cart_pos_x, m_cart_pos_y - y_offset - (m_cart_height / 2.0f)));
	//pole.setPosition(sf::Vector2f(m_cart_pos_x, 250));
	//pole.setOrigin(sf::Vector2f(m_pole_thickness / 2.0f, m_pole_height / 2));
	//window.draw(cart);
	//window.draw(pole);
	//pole.setOrigin(sf::Vector2f(-m_cart_pos_x, -m_cart_pos_y + m_cart_height + m_pole_height / 2.0f));
	pole.setPosition(sf::Vector2f(m_cart_pos_x, m_cart_pos_y - m_cart_height - m_pole_height / 2.0f));
	pole.setRotation(m_angle * (180 / 3.14159265358979323));
	window.draw(pole);
}

void CartPole::calculate_physics() {
	float c_time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;

	m_angular_acceleration = m_gravity * std::sinf(m_angle) + std::cosf(m_angle) * ((-m_force - m_pole_mass * m_pole_length * m_angular_velocity * m_angular_velocity * std::sinf(m_angle)) / (m_cart_mass + m_pole_mass));
	m_angular_acceleration /= m_pole_length * (1.33f - ((m_pole_mass * std::cosf(m_angle) * std::cosf(m_angle)) / (m_pole_mass + m_cart_mass)));

	m_cart_normal = (m_cart_mass + m_pole_mass) * m_gravity - m_pole_mass * m_pole_length * (m_angular_acceleration * std::sinf(m_angle) + m_angular_velocity * m_angular_velocity * std::cosf(m_angle));

	m_cart_acceleration = (m_force + m_pole_mass * m_pole_length * (m_angular_velocity * m_angular_velocity * std::sinf(m_angle) - m_angular_acceleration * std::cosf(m_angle))) / (m_cart_mass + m_pole_mass);

	m_angular_velocity += m_angular_acceleration * (c_time - time);
	m_angle += m_angular_velocity * (c_time - time);
	printf("%f\t%f\t%f\n", m_angular_acceleration, m_angular_velocity, m_angle);

	m_cart_velocity += m_cart_acceleration * m_cart_velocity_scale * (c_time - time);
	m_cart_pos_x += m_cart_velocity * m_cart_velocity_scale * (c_time - time);
	printf("%f\t%f\t%f\n\n", m_cart_acceleration, m_cart_velocity, m_cart_pos_x);

	time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;
}

void CartPole::key_pressed_handler(sf::Event key_event) {
	if (key_event.key.scancode == sf::Keyboard::Scan::D)
		m_force = 1;
	if (key_event.key.scancode == sf::Keyboard::Scan::A)
		m_force = -1;
}

void CartPole::key_released_handler(sf::Event key_event) {
	if (key_event.key.scancode == sf::Keyboard::Scan::D)
		m_force = 0;
	if (key_event.key.scancode == sf::Keyboard::Scan::A)
		m_force = 0;
}

void CartPole::join_window() {
	main_loop_thd.join();
}