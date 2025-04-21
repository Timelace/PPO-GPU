#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CarGame {
public:
	CarGame(float width, float height, float car_width, float car_height);
	__device__ void update_wheel_velocities(float wheel_1_vel, float wheel_2_vel, float wheel_3_vel, float wheel_4_vel);
	__device__ void timestep();
private:
	const float m_width, m_height, m_car_width, m_car_height;

	float m_wheel_1_vel = 0, m_wheel_2_vel = 0, m_wheel_3_vel = 0, m_wheel_4_vel = 0;

	const float m_forward_percent = 0.002, m_angle_percent = 0.0005;
	const float m_forward, m_angle;

	float m_center_x, m_center_y;
	float m_angle_car;

	float m_target_x, m_target_y, m_target_r = 10;
	float m_score = 0;
};
