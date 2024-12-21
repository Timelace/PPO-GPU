#include "SnakeGame.h"
#include <random>
#include <ctime>
#include <iostream>

SnakeGame::SnakeGame(float width, float height, int tiles_x, int tiles_y, float framerate) :
	m_width(width),
	m_height(height),
	m_tiles_x(tiles_x),
	m_tiles_y(tiles_y),
	m_framerate(framerate)
{
	bg_squares = (sf::Vertex*)malloc((tiles_x + tiles_y) * 2 * sizeof(sf::Vertex));

	std::srand((std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count());
	int rand_x = ((int)((rand() / (float)RAND_MAX) * tiles_x)), rand_y = ((int)((rand() / (float)RAND_MAX) * tiles_y));
	apple_pos[0] = rand_x * (m_width / tiles_x);
	apple_pos[1] = rand_y * (m_height / tiles_y);
	apple.setFillColor(sf::Color(240, 72, 91));
	apple.setPosition(sf::Vector2f(apple_pos[0], apple_pos[1]));
	apple.setSize(sf::Vector2f((m_width / m_tiles_x), (m_height / m_tiles_y)));

	time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;

	sf::RectangleShape* head_segment;
	head_segment = new sf::RectangleShape(sf::Vector2f((m_width / m_tiles_x), (m_height / m_tiles_y)));
	head_segment->setFillColor(sf::Color(7, 219, 17));
	snake_body.push_back(*head_segment);
	snake_body[0].setPosition(sf::Vector2f((m_width / tiles_x) * (tiles_x / 2.0f), (m_height / tiles_y) * (tiles_y / 2.0f)));
	snake_dirs.push_back(Direction::Right);

	start_direction_prompt_font.loadFromFile("E:/Machine_Learning_Cpp/temporary/PPO-GPU/x64/Fonts/toxigenesis bd.ttf");
	start_direction_prompt.setFont(start_direction_prompt_font);
	start_direction_prompt.setString("\t\t\tPress\nWASD / Arrow Keys\n\t\t  To Begin");
	start_direction_prompt.setPosition(width * 0.05, height * 0.35);
	start_direction_prompt.setCharacterSize((width > height) ? height * 0.08 : width * 0.08);
	start_direction_prompt.setColor(sf::Color(92, 92, 92));

	main_loop_thread = std::thread(&SnakeGame::main_loop, this);
}

SnakeGame::~SnakeGame() {
	window.close();
	free(bg_squares);
}

void SnakeGame::main_loop() {
	window.create(sf::VideoMode(m_width, m_height), "Snake SNAKE");
	window.setVerticalSyncEnabled(true);
	game_view.reset(sf::FloatRect(0, 0, m_width, m_height));
	window.setView(game_view);
	sf::Event events;
	while (window.isOpen()) {
		while (window.pollEvent(events)) {
			if (events.type == sf::Event::Closed)
				window.close();
			if (events.type == sf::Event::KeyPressed)
				keypress_event(events.key.scancode);
			if (events.type == sf::Event::MouseWheelScrolled)
				scroll_event(events.mouseWheelScroll.delta);
		}
		if (!m_is_dead) {
			if (m_dir != Direction::None) {
				if (((std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f) - time >= m_framerate) {
					time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;
					update_snake_pos();
					check_snake_life();
				}
			}
			draw_current_state();
		}
		else {
			if (!played_death_animation) {
				animate_snake_death();
				setup_death_screen();
				draw_empty_state();
				animate_death_screen();
				played_death_animation = true;
			}
			draw_empty_state();
			draw_end_screen();
		}
		window.display();
	}
}

void SnakeGame::draw_current_state() {
	window.clear(sf::Color::Black);
	for (int x = 0; x < m_tiles_x; x++) {
		bg_squares[x * 2].position = sf::Vector2f((m_width / (m_tiles_x)) * x, 0);
		bg_squares[x * 2].color = sf::Color(127, 128, 138);
		bg_squares[x * 2 + 1].position = sf::Vector2f((m_width / (m_tiles_x)) * x, m_height);
		bg_squares[x * 2 + 1].color = sf::Color(127, 128, 138);
	}
	for (int y = 0; y < m_tiles_y; y++) {
		bg_squares[(m_tiles_x + y) * 2].position = sf::Vector2f(0, (m_height / (m_tiles_y)) * (y + 1));
		bg_squares[(m_tiles_x + y) * 2].color = sf::Color(127, 128, 138);
		bg_squares[(m_tiles_x + y) * 2 + 1].position = sf::Vector2f(m_width, (m_height / (m_tiles_y)) * (y + 1));
		bg_squares[(m_tiles_x + y) * 2 + 1].color = sf::Color(127, 128, 138);
	}
	if(m_dir == Direction::None)
		window.draw(start_direction_prompt);
	apple.setPosition(sf::Vector2f(apple_pos[0], apple_pos[1]));
	for (sf::RectangleShape snake_segment : snake_body) {
		window.draw(snake_segment);
	}
	window.draw(apple);
	window.draw(bg_squares, (m_tiles_x + m_tiles_y) * 2, sf::Lines);
}

void SnakeGame::update_snake_pos() {
	// update head
	int final_part_index = 0;
	if (m_dir != snake_dirs[snake_dirs.size() - 1]) {
		final_part_index = snake_dirs.size();
		float last_x = snake_body[final_part_index - 1].getPosition().x, last_y = snake_body[final_part_index - 1].getPosition().y;
		if (snake_dirs[final_part_index - 1] == Direction::Right) {
			last_x += snake_body[final_part_index - 1].getSize().x - (m_width / m_tiles_x);
		}
		if (snake_dirs[final_part_index - 1] == Direction::Down) {
			last_y += snake_body[final_part_index - 1].getSize().y - (m_height / m_tiles_y);
		}
		sf::RectangleShape* new_segment;
		new_segment = new sf::RectangleShape(sf::Vector2f((m_width / m_tiles_x), (m_height / m_tiles_y)));
		new_segment->setFillColor(sf::Color(7, 219, 17));
		snake_body.push_back(*new_segment);
		snake_dirs.push_back(m_dir);
		switch (m_dir) {
		case Direction::Up:
			snake_body[final_part_index].setPosition(sf::Vector2f(last_x, last_y - (m_height / m_tiles_y)));
			break;
		case Direction::Right:
			snake_body[final_part_index].setPosition(sf::Vector2f(last_x + (m_width / m_tiles_x), last_y));
			break;
		case Direction::Down:
			snake_body[final_part_index].setPosition(sf::Vector2f(last_x, last_y + (m_height / m_tiles_y)));
			break;
		case Direction::Left:
			snake_body[final_part_index].setPosition(sf::Vector2f(last_x - (m_width / m_tiles_x), last_y));
			break;
		}
	}
	else {
		final_part_index = snake_dirs.size() - 1;
		switch (m_dir) {
		case Direction::Up:
			snake_body[final_part_index].setSize(sf::Vector2f(snake_body[final_part_index].getSize().x, snake_body[final_part_index].getSize().y + (m_height / m_tiles_y)));
			snake_body[final_part_index].setPosition(sf::Vector2f(snake_body[final_part_index].getPosition().x, snake_body[final_part_index].getPosition().y - (m_height / m_tiles_y)));
			break;
		case Direction::Right:
			snake_body[final_part_index].setSize(sf::Vector2f(snake_body[final_part_index].getSize().x + (m_width / m_tiles_x), snake_body[final_part_index].getSize().y));
			break;
		case Direction::Down:
			snake_body[final_part_index].setSize(sf::Vector2f(snake_body[final_part_index].getSize().x, snake_body[final_part_index].getSize().y + (m_height / m_tiles_y)));
			break;
		case Direction::Left:
			snake_body[final_part_index].setSize(sf::Vector2f(snake_body[final_part_index].getSize().x + (m_width / m_tiles_x), snake_body[final_part_index].getSize().y));
			snake_body[final_part_index].setPosition(sf::Vector2f(snake_body[final_part_index].getPosition().x - (m_width / m_tiles_x), snake_body[final_part_index].getPosition().y));
			break;
		}
	}

	// eliminate last part of tail
	if (pause_tail_elim_frames > 0) {
		pause_tail_elim_frames--;
	}
	else {
		Direction tail_dir = snake_dirs[0];
		float new_width = 0, new_height = 0;
		switch (tail_dir) {
		case Direction::Up:
			new_height = snake_body[0].getSize().y - (m_height / m_tiles_y);
			if (new_height < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
				snake_body.erase(snake_body.begin());
				snake_dirs.erase(snake_dirs.begin());
			}
			else {
				snake_body[0].setSize(sf::Vector2f(snake_body[0].getSize().x, new_height));
			}
			break;
		case Direction::Right:
			new_width = snake_body[0].getSize().x - (m_width / m_tiles_x);
			if (new_width < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
				snake_body.erase(snake_body.begin());
				snake_dirs.erase(snake_dirs.begin());
			}
			else {
				snake_body[0].setSize(sf::Vector2f(new_width, snake_body[0].getSize().y));
				snake_body[0].setPosition(sf::Vector2f(snake_body[0].getPosition().x + (m_width / m_tiles_x), snake_body[0].getPosition().y));
			}
			break;
		case Direction::Down:
			new_height = snake_body[0].getSize().y - (m_height / m_tiles_y);
			if (new_height < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
				snake_body.erase(snake_body.begin());
				snake_dirs.erase(snake_dirs.begin());
			}
			else {
				snake_body[0].setSize(sf::Vector2f(snake_body[0].getSize().x, new_height));
				snake_body[0].setPosition(sf::Vector2f(snake_body[0].getPosition().x, snake_body[0].getPosition().y + (m_height / m_tiles_y)));
			}
			break;
		case Direction::Left:
			new_width = snake_body[0].getSize().x - (m_width / m_tiles_x);
			if (new_width < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
				snake_body.erase(snake_body.begin());
				snake_dirs.erase(snake_dirs.begin());
			}
			else {
				snake_body[0].setSize(sf::Vector2f(new_width, snake_body[0].getSize().y));
			}
			break;
		}
	}


	// on eat apple
	float actual_x = snake_body[final_part_index].getPosition().x, actual_y = snake_body[final_part_index].getPosition().y;
	if (m_dir == Direction::Right) {
		actual_x += snake_body[final_part_index].getSize().x - (m_width / m_tiles_x);
	}
	if (m_dir == Direction::Down) {
		actual_y += snake_body[final_part_index].getSize().y - (m_width / m_tiles_x);
	}
	if (actual_x == apple.getPosition().x && actual_y == apple.getPosition().y) {
		pause_tail_elim_frames += size_per_apple;
		m_score++;
		std::srand((std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count());
		int rand_x = ((int)((rand() / (float)RAND_MAX) * m_tiles_x)), rand_y = ((int)((rand() / (float)RAND_MAX) * m_tiles_y));
		apple_pos[0] = rand_x * (m_width / m_tiles_x);
		apple_pos[1] = rand_y * (m_height / m_tiles_y);
		apple.setPosition(sf::Vector2f(apple_pos[0], apple_pos[1]));
	}

}

void SnakeGame::check_snake_life() {
	// check for self-intersection
	float c_x = snake_body[snake_body.size() - 1].getPosition().x, c_y = snake_body[snake_body.size() - 1].getPosition().y;
	if (m_dir == Direction::Right) {
		c_x += snake_body[snake_body.size() - 1].getSize().x - (m_width / m_tiles_x);
	}
	if (m_dir == Direction::Down) {
		c_y += snake_body[snake_body.size() - 1].getSize().y - (m_width / m_tiles_x);
	}
	for (int i = snake_body.size() - 3; i >= 0; i--) {
		if (snake_dirs[i] == Direction::Left || snake_dirs[i] == Direction::Right) {
			if (c_x >= snake_body[i].getPosition().x && c_x <= (snake_body[i].getPosition().x + snake_body[i].getSize().x) &&
				c_y == snake_body[i].getPosition().y)
			{
				m_is_dead = true;
				return;
			}
		}
		if (snake_dirs[i] == Direction::Up || snake_dirs[i] == Direction::Down) {
			if (c_x == snake_body[i].getPosition().x &&
				c_y >= snake_body[i].getPosition().y && c_y <= (snake_body[i].getPosition().y + snake_body[i].getSize().y))
			{
				m_is_dead = true;
				return;
			}
		}
	}
	// check for out of bounds
	if (c_x < 0 || c_x > m_width || c_y < 0 || c_y > m_height) {
		m_is_dead = true;
		return;
	}
}

void SnakeGame::draw_empty_state() {
	window.clear(sf::Color::Black);
	for (int x = 0; x < m_tiles_x; x++) {
		bg_squares[x * 2].position = sf::Vector2f((m_width / (m_tiles_x)) * x, 0);
		bg_squares[x * 2].color = sf::Color(127, 128, 138);
		bg_squares[x * 2 + 1].position = sf::Vector2f((m_width / (m_tiles_x)) * x, m_height);
		bg_squares[x * 2 + 1].color = sf::Color(127, 128, 138);
	}
	for (int y = 0; y < m_tiles_y; y++) {
		bg_squares[(m_tiles_x + y) * 2].position = sf::Vector2f(0, (m_height / (m_tiles_y)) * (y + 1));
		bg_squares[(m_tiles_x + y) * 2].color = sf::Color(127, 128, 138);
		bg_squares[(m_tiles_x + y) * 2 + 1].position = sf::Vector2f(m_width, (m_height / (m_tiles_y)) * (y + 1));
		bg_squares[(m_tiles_x + y) * 2 + 1].color = sf::Color(127, 128, 138);
	}
	window.draw(bg_squares, (m_tiles_x + m_tiles_y) * 2, sf::Lines);
}

void SnakeGame::draw_end_screen() {
	for (sf::RectangleShape pixel : end_screen_death_text) {
		window.draw(pixel);
	}
}

void SnakeGame::setup_death_screen() {
	// y letter
	int tile_coord_x_0 = (int)(0.3 * m_tiles_x);
	int tile_coord_y_0 = (int)(0.1 * m_tiles_y);
	int tile_coord_x_1 = (int)(0.35 * m_tiles_x);
	int tile_coord_y_1 = (int)(0.2 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.35 * m_tiles_x);
	tile_coord_y_0 = (int)(0.2 * m_tiles_y);
	tile_coord_x_1 = (int)(0.4 * m_tiles_x);
	tile_coord_y_1 = (int)(0.1 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.35 * m_tiles_x);
	tile_coord_y_0 = (int)(0.2 * m_tiles_y);
	tile_coord_x_1 = (int)(0.35 * m_tiles_x);
	tile_coord_y_1 = (int)(0.3 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);

	// o letter
	tile_coord_x_0 = (int)(0.45 * m_tiles_x);
	tile_coord_y_0 = (int)(0.1 * m_tiles_y);
	tile_coord_x_1 = (int)(0.45 * m_tiles_x);
	tile_coord_y_1 = (int)(0.3 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.45 * m_tiles_x);
	tile_coord_y_0 = (int)(0.3 * m_tiles_y);
	tile_coord_x_1 = (int)(0.55 * m_tiles_x);
	tile_coord_y_1 = (int)(0.3 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.55 * m_tiles_x);
	tile_coord_y_0 = (int)(0.3 * m_tiles_y);
	tile_coord_x_1 = (int)(0.55 * m_tiles_x);
	tile_coord_y_1 = (int)(0.1 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.55 * m_tiles_x);
	tile_coord_y_0 = (int)(0.1 * m_tiles_y);
	tile_coord_x_1 = (int)(0.45 * m_tiles_x);
	tile_coord_y_1 = (int)(0.1 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);

	// u letter
	tile_coord_x_0 = (int)(0.6 * m_tiles_x);
	tile_coord_y_0 = (int)(0.1 * m_tiles_y);
	tile_coord_x_1 = (int)(0.6 * m_tiles_x);
	tile_coord_y_1 = (int)(0.3 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.6 * m_tiles_x);
	tile_coord_y_0 = (int)(0.3 * m_tiles_y);
	tile_coord_x_1 = (int)(0.7 * m_tiles_x);
	tile_coord_y_1 = (int)(0.3 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.7 * m_tiles_x);
	tile_coord_y_0 = (int)(0.3 * m_tiles_y);
	tile_coord_x_1 = (int)(0.7 * m_tiles_x);
	tile_coord_y_1 = (int)(0.1 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);

	// d letter
	tile_coord_x_0 = (int)(0.225 * m_tiles_x);
	tile_coord_y_0 = (int)(0.35 * m_tiles_y);
	tile_coord_x_1 = (int)(0.225 * m_tiles_x);
	tile_coord_y_1 = (int)(0.55 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.225 * m_tiles_x);
	tile_coord_y_0 = (int)(0.55 * m_tiles_y);
	tile_coord_x_1 = (int)(0.25 * m_tiles_x);
	tile_coord_y_1 = (int)(0.55 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.25 * m_tiles_x);
	tile_coord_y_0 = (int)(0.55 * m_tiles_y);
	tile_coord_x_1 = (int)(0.325 * m_tiles_x);
	tile_coord_y_1 = (int)(0.45 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.325 * m_tiles_x);
	tile_coord_y_0 = (int)(0.45 * m_tiles_y);
	tile_coord_x_1 = (int)(0.25 * m_tiles_x);
	tile_coord_y_1 = (int)(0.35 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.25 * m_tiles_x);
	tile_coord_y_0 = (int)(0.35 * m_tiles_y);
	tile_coord_x_1 = (int)(0.225 * m_tiles_x);
	tile_coord_y_1 = (int)(0.35 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);

	// i letter
	tile_coord_x_0 = (int)(0.425 * m_tiles_x);
	tile_coord_y_0 = (int)(0.35 * m_tiles_y);
	tile_coord_x_1 = (int)(0.425 * m_tiles_x);
	tile_coord_y_1 = (int)(0.55 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);

	// e letter
	tile_coord_x_0 = (int)(0.525 * m_tiles_x);
	tile_coord_y_0 = (int)(0.35 * m_tiles_y);
	tile_coord_x_1 = (int)(0.525 * m_tiles_x);
	tile_coord_y_1 = (int)(0.55 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.525 * m_tiles_x);
	tile_coord_y_0 = (int)(0.35 * m_tiles_y);
	tile_coord_x_1 = (int)(0.625 * m_tiles_x);
	tile_coord_y_1 = (int)(0.35 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.525 * m_tiles_x);
	tile_coord_y_0 = (int)(0.45 * m_tiles_y);
	tile_coord_x_1 = (int)(0.625 * m_tiles_x);
	tile_coord_y_1 = (int)(0.45 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.525 * m_tiles_x);
	tile_coord_y_0 = (int)(0.55 * m_tiles_y);
	tile_coord_x_1 = (int)(0.625 * m_tiles_x);
	tile_coord_y_1 = (int)(0.55 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);

	// d letter 2
	tile_coord_x_0 = (int)(0.675 * m_tiles_x);
	tile_coord_y_0 = (int)(0.35 * m_tiles_y);
	tile_coord_x_1 = (int)(0.675 * m_tiles_x);
	tile_coord_y_1 = (int)(0.55 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.675 * m_tiles_x);
	tile_coord_y_0 = (int)(0.55 * m_tiles_y);
	tile_coord_x_1 = (int)(0.700 * m_tiles_x);
	tile_coord_y_1 = (int)(0.55 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.700 * m_tiles_x);
	tile_coord_y_0 = (int)(0.55 * m_tiles_y);
	tile_coord_x_1 = (int)(0.775 * m_tiles_x);
	tile_coord_y_1 = (int)(0.45 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.775 * m_tiles_x);
	tile_coord_y_0 = (int)(0.45 * m_tiles_y);
	tile_coord_x_1 = (int)(0.700 * m_tiles_x);
	tile_coord_y_1 = (int)(0.35 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
	tile_coord_x_0 = (int)(0.700 * m_tiles_x);
	tile_coord_y_0 = (int)(0.35 * m_tiles_y);
	tile_coord_x_1 = (int)(0.675 * m_tiles_x);
	tile_coord_y_1 = (int)(0.35 * m_tiles_y);
	draw_line(tile_coord_x_0, tile_coord_y_0, tile_coord_x_1, tile_coord_y_1);
}

void SnakeGame::animate_snake_death() {
	float time_per_size_death = total_death_time / (1 + m_score * size_per_apple);
	while (snake_body.size() != 0) {
		if ((std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f - time >= time_per_size_death) {
			time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;
			int final_index = snake_body.size() - 1;
			Direction tail_dir = snake_dirs[final_index];
			float new_width = 0, new_height = 0;
			switch (tail_dir) {
			case Direction::Up:
				new_height = snake_body[final_index].getSize().y - (m_height / m_tiles_y);
				if (new_height < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
					snake_body.erase(snake_body.begin() + final_index);
					snake_dirs.erase(snake_dirs.begin() + final_index);
				}
				else {
					snake_body[final_index].setSize(sf::Vector2f(snake_body[final_index].getSize().x, new_height));
					snake_body[final_index].setPosition(sf::Vector2f(snake_body[final_index].getPosition().x, snake_body[final_index].getPosition().y + (m_height / m_tiles_y)));
				}
				break;
			case Direction::Right:
				new_width = snake_body[final_index].getSize().x - (m_width / m_tiles_x);
				if (new_width < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
					snake_body.erase(snake_body.begin() + final_index);
					snake_dirs.erase(snake_dirs.begin() + final_index);
				}
				else {
					snake_body[final_index].setSize(sf::Vector2f(new_width, snake_body[final_index].getSize().y));
				}
				break;
			case Direction::Down:
				new_height = snake_body[final_index].getSize().y - (m_height / m_tiles_y);
				if (new_height < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
					snake_body.erase(snake_body.begin() + final_index);
					snake_dirs.erase(snake_dirs.begin() + final_index);
				}
				else {
					snake_body[final_index].setSize(sf::Vector2f(snake_body[final_index].getSize().x, new_height));
				}
				break;
			case Direction::Left:
				new_width = snake_body[final_index].getSize().x - (m_width / m_tiles_x);
				if (new_width < (m_height / m_tiles_y)) { // delete final segment if we dont need it anymore
					snake_body.erase(snake_body.begin() + final_index);
					snake_dirs.erase(snake_dirs.begin() + final_index);
				}
				else {
					snake_body[final_index].setSize(sf::Vector2f(new_width, snake_body[final_index].getSize().y));
					snake_body[final_index].setPosition(sf::Vector2f(snake_body[final_index].getPosition().x + (m_width / m_tiles_x), snake_body[final_index].getPosition().y));
				}
				break;
			}
		}
		draw_current_state();
		window.display();
	}
}

void SnakeGame::animate_death_screen() {
	std::vector<int> points(end_screen_death_text.size());
	std::vector<int> seen_points(end_screen_death_text.size());
	int seen_points_index = 0;
	for (int i = 0; i < end_screen_death_text.size(); i++)
		points[i] = i;
	int batches = 5;
	int elems_per_batch = end_screen_death_text.size() / batches;
	draw_empty_state();
	while (points.size() > 0) {
		if (((std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f) - time >= 0.5) {
			time = (std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now()).time_since_epoch())).count() / 1000.0f;
			int elems = (elems_per_batch > points.size()) ? points.size() : elems_per_batch;
			for (int e = 0; e < elems; e++) {
				int index = (std::rand() / (float)RAND_MAX) * points.size();
				window.draw(end_screen_death_text[points[index]]);
				seen_points[seen_points_index] = points[index];
				seen_points_index++;
				points.erase(points.begin() + index);
			}
			for (int i = 0; i < seen_points_index; i++) {
				window.draw(end_screen_death_text[seen_points[i]]);
			}
			window.display();
		}
	}
}

void SnakeGame::draw_line(int x_0, int y_0, int x_1, int y_1) {
	int d_x = std::abs(x_1 - x_0);
	int s_x = x_0 < x_1 ? 1 : -1;
	int d_y = -std::abs(y_1 - y_0);
	int s_y = y_0 < y_1 ? 1 : -1;
	int error = d_x + d_y;

	while (true) {
		sf::RectangleShape* pixel = new sf::RectangleShape(sf::Vector2f((m_width / m_tiles_x), (m_height / m_tiles_y)));
		pixel->setPosition(sf::Vector2f(x_0 * (m_width / m_tiles_x), y_0 * (m_height / m_tiles_y)));
		pixel->setFillColor(sf::Color(7, 219, 17));
		end_screen_death_text.push_back(*pixel);
		if (x_0 == x_1 && y_0 == y_1)
			break;
		int e_2 = 2 * error;
		if (e_2 >= d_y) {
			error += d_y;
			x_0 += s_x;
		}
		if (e_2 <= d_x) {
			error += d_x;
			y_0 += s_y;
		}
	}
}


void SnakeGame::keypress_event(sf::Keyboard::Scancode key) {
	if (key == sf::Keyboard::Scan::W || key == sf::Keyboard::Scan::Up) {
		m_dir = Direction::Up;
		return;
	}
	if (key == sf::Keyboard::Scan::D || key == sf::Keyboard::Scan::Right) {
		m_dir = Direction::Right;
		return;
	}
	if (key == sf::Keyboard::Scan::S || key == sf::Keyboard::Scan::Down) {
		m_dir = Direction::Down;
		return;
	}
	if (key == sf::Keyboard::Scan::A || key == sf::Keyboard::Scan::Left) {
		m_dir = Direction::Left;
		return;
	}

}

void SnakeGame::scroll_event(float distance) {
	float framerate_factor = (distance > 0) ? m_framerate_scroll_factor : m_framerate_scroll_factor + 1;
	m_framerate *= framerate_factor;
}