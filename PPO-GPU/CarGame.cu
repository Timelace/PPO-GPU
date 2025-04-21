#include "CarGame.h"


#include <iostream>

CarGame::CarGame(float width, float height, float car_width, float car_height) :
	m_width(width),
	m_height(height),
	m_car_width(car_width),
	m_car_height(car_height),
	m_forward(m_forward_percent * height),
	m_angle(m_angle_percent * width)
{
	m_center_x = width / 2;
	m_center_y = height / 2;

	m_angle_car = 90;

	std::srand(std::time(0));

	m_target_x = (std::rand() / (float)RAND_MAX) * 900;
	m_target_y = (std::rand() / (float)RAND_MAX) * 900;
}

__device__ void CarGame::update_wheel_velocities(float wheel_1_vel, float wheel_2_vel, float wheel_3_vel, float wheel_4_vel) {
	m_wheel_1_vel = wheel_1_vel;
	m_wheel_2_vel = wheel_2_vel;
	m_wheel_3_vel = wheel_3_vel;
	m_wheel_4_vel = wheel_4_vel;
}

__device__ void CarGame::timestep() {
	float total_movement = m_wheel_1_vel * m_forward + m_wheel_2_vel * m_forward + m_wheel_3_vel * m_forward + m_wheel_4_vel * m_forward;
	float x_change = -std::cosf(m_angle_car * (3.1415926 / 180.0f)) * total_movement;
	m_center_x += x_change;
	m_center_y += std::tanf(m_angle_car * (3.1415926 / 180)) * x_change;

	m_angle_car += m_wheel_1_vel * -m_angle + m_wheel_2_vel * m_angle + m_wheel_3_vel * -m_angle + m_wheel_4_vel * m_angle;
	if (m_angle_car > 360)
		m_angle_car -= 360;
	if (m_angle_car < 0)
		m_angle_car += 360;

	if (std::sqrt((m_center_x - m_target_x) * (m_center_x - m_target_x) + (m_center_y - m_target_y) * (m_center_y - m_target_y)) < m_target_r) {
		m_score++;
		m_target_x = (std::rand() / (float)RAND_MAX) * 900;
		m_target_y = (std::rand() / (float)RAND_MAX) * 900;
	}
}