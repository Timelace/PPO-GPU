#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

class SnakeGame {
public:
	SnakeGame(float width, float height, int tiles_x, int tiles_y, float framerate);
	~SnakeGame();
private:
	sf::RenderWindow window;
	sf::View game_view;
	sf::Vertex* bg_squares;
	sf::RectangleShape apple;

	sf::Font start_direction_prompt_font;
	sf::Text start_direction_prompt;

	float time;
	float m_width, m_height;
	const int m_tiles_x, m_tiles_y;
	float m_framerate;
	float m_framerate_scroll_factor = 0.5;

	std::thread main_loop_thread;
	void main_loop();

	int m_score = 0;

	enum Direction {
		Up, Down, Left, Right, None
	};

	bool m_is_dead = false;
	bool played_death_animation = false;
	std::vector<sf::RectangleShape> snake_body;
	std::vector<Direction> snake_dirs;
	float apple_pos[2];
	int pause_tail_elim_frames = 0;
	int size_per_apple = 10;

	Direction m_dir = Direction::None;

	void draw_current_state();
	void update_snake_pos();
	void check_snake_life();

	void draw_end_screen();
	void draw_empty_state();
	void setup_death_screen();
	void animate_snake_death();
	float total_death_time = 3.0f;
	void animate_death_screen();
	void draw_line(int x_0, int y_0, int x_1, int y_1);
	std::vector<sf::RectangleShape> end_screen_death_text;

	void keypress_event(sf::Keyboard::Scancode key);
	void scroll_event(float distance);

};
