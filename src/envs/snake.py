from src.envs.env import ParallelEnvironment
from src.utils.utils import bool2num, num2bool

import numpy as np
import pygame

FIELD_SZ = 30

STYLE = {"bg_color": (50, 50, 50),
         "field_color": (70, 70, 70),
         "border_color": (130, 130, 130),
         "snake_color": (170, 240, 30),
         "fruit_color": (240, 170, 30),
         "font_color": (200, 200, 200)}

MAX_TIME_WITHOUT_SCORE = 288  # to avoid infinite loops 288

STD_HEIGHT = 12
STD_WIDTH = 24


class Snake(ParallelEnvironment):
    NAME = "snake"
    LEVELS = False
    TIME_RELEVANT = True
    WINS_RELEVANT = False

    def __init__(self, num_par_inst, height=STD_HEIGHT, width=STD_WIDTH):
        actions = ['MOVE_UPWARDS', 'MOVE_RIGHTWARDS', 'MOVE_DOWNWARDS', 'MOVE_LEFTWARDS']
        super().__init__(num_par_inst, actions)

        self.height = height
        self.width = width

        # Constants
        self.coordinates = np.moveaxis(np.mgrid[0:height, 0:width], 0, 2)
        self.max_score = self.height * self.width

        # Locations
        self.snake_head_locations = np.zeros(shape=(self.num_par_inst, 2), dtype="int8")
        self.snake_bodies = SnakeBodyBuffer(length=self.height * self.width, width=self.num_par_inst)
        self.fruit_locations = np.zeros(shape=(self.num_par_inst, 2), dtype="uint8")

        # 0: top, 1: right, 2: bottom, 3: left
        self.snake_movement_orientations = np.zeros(shape=self.num_par_inst, dtype="uint8")

        # Fields
        self.snake_head_fields = np.zeros(shape=(self.num_par_inst, self.height, self.width), dtype="bool")
        self.snake_body_fields = np.zeros(shape=(self.num_par_inst, self.height, self.width), dtype="bool")
        self.fruit_fields = np.zeros(shape=(self.num_par_inst, self.height, self.width), dtype="bool")

        # State control
        self.time_since_last_fruit = np.zeros(shape=self.num_par_inst, dtype="uint")

        self.screen = None

        self.__init_env(range(self.num_par_inst))

    def reset(self):
        self.snake_head_locations[:] = 0
        self.snake_bodies.reset()
        self.fruit_locations[:] = 0

        self.snake_movement_orientations[:] = 0

        self.snake_head_fields[:] = False
        self.snake_body_fields[:] = False
        self.fruit_fields[:] = False

        self.scores[:] = 0
        self.times[:] = 0
        self.game_overs[:] = False
        self.time_since_last_fruit[:] = 0

        self.__init_env(range(self.num_par_inst))

    def reset_for(self, ids):
        self.snake_head_locations[ids] = 0
        self.snake_bodies.reset_for(ids)
        self.fruit_locations[ids] = 0

        self.snake_movement_orientations[ids] = 0

        self.snake_head_fields[ids] = False
        self.snake_body_fields[ids] = False
        self.fruit_fields[ids] = False

        self.scores[ids] = 0
        self.times[ids] = 0
        self.game_overs[ids] = False
        self.time_since_last_fruit[ids] = 0

        self.__init_env(ids)

    def __init_env(self, ids):
        # Initialize snake head (with margin to border)
        self.snake_head_locations[ids] = np.random.randint(low=(1, 1), high=(self.height - 1, self.width - 1),
                                                           size=(len(ids), 2))
        self.snake_bodies.init_head_locations(ids, self.snake_head_locations[ids])

        # Initialize movement orientation
        self.snake_movement_orientations[ids] = np.random.randint(low=0, high=4, size=len(ids))

        # Initialize fruit
        self.spawn_fruit(ids)

        # Initialize fields
        self.compute_fields()

    def spawn_fruit(self, ids):
        self.fruit_fields[ids] = False
        for idx in ids:
            valid_spawn_coords = self.coordinates[~ self.snake_body_fields[idx]]
            spawn_field_id = np.random.randint(0, len(valid_spawn_coords))
            spawn_coords = valid_spawn_coords[spawn_field_id]
            self.fruit_locations[idx] = spawn_coords
            self.fruit_fields[idx, spawn_coords[0], spawn_coords[1]] = True

        # self.fruit_locations[ids] = np.random.randint(low=(0, 0), high=(self.height, self.width), size=(len(ids), 2))
        # self.fruit_fields[ids] = False
        # self.fruit_fields[ids, self.fruit_locations[ids, 0], self.fruit_locations[ids, 1]] = True

    def step(self, actions):
        # Update the env
        self.update_snake_movement_orientation(actions)
        fruit_found = self.snake_creep()
        self.snake_bodies.grow(fruit_found)
        self.spawn_fruit(np.where(fruit_found)[0])

        # Compute statistics
        self.times[~ self.game_overs] += 1
        self.time_since_last_fruit += 1
        self.scores = self.snake_bodies.get_lengths()

        rewards = np.zeros(self.num_par_inst)
        rewards[fruit_found] = 1  # np.log2((self.scores[fruit_found] + 2) / (self.scores[fruit_found] + 1))
        # - self.time_since_last_score / MAX_TIME_WITHOUT_SCORE
        # Encourage faster fruit gathering: doesn't work
        # rewards[:] -= 0.005  # rotten fruit

        game_won = self.scores == self.max_score
        rewards[game_won] = 10
        self.game_overs[game_won] = True

        self.time_since_last_fruit[fruit_found] = 0
        starved = self.time_since_last_fruit >= MAX_TIME_WITHOUT_SCORE
        self.game_overs[starved] = True

        rewards[self.game_overs] -= 1

        return rewards, self.scores, self.game_overs, self.times, self.wins

    def update_snake_movement_orientation(self, actions):
        action_valid = (self.snake_movement_orientations - actions) % 2 != 0
        self.snake_movement_orientations[action_valid] = actions[action_valid]

    def snake_creep(self):
        self.move_snake_tail()
        self.move_snake_head()

        game_overs, fruit_found = self.check_for_collisions()  # 1 ms
        self.game_overs[:] = game_overs

        self.compute_fields()

        return fruit_found

    def move_snake_tail(self):
        tail_locations = self.snake_bodies.get_tail_locations()
        self.snake_body_fields[range(self.num_par_inst), tail_locations[:, 0], tail_locations[:, 1]] = False

    def move_snake_head(self):
        """Moves snake head in direction of orientation"""
        top = self.snake_movement_orientations == 0
        right = self.snake_movement_orientations == 1
        bottom = self.snake_movement_orientations == 2
        left = self.snake_movement_orientations == 3
        self.snake_head_locations[top, 0] -= 1
        self.snake_head_locations[right, 1] += 1
        self.snake_head_locations[bottom, 0] += 1
        self.snake_head_locations[left, 1] -= 1

        self.snake_bodies.move_heads(self.snake_head_locations)

    def check_for_collisions(self):
        snake_heads_y = self.snake_head_locations[:, 0]
        snake_heads_x = self.snake_head_locations[:, 1]

        collides_top_wall = snake_heads_y < 0
        collides_right_wall = snake_heads_x >= self.width
        collides_bottom_wall = snake_heads_y >= self.height
        collides_left_wall = snake_heads_x < 0
        collides_wall = collides_top_wall | collides_right_wall | collides_bottom_wall | collides_left_wall

        collides_self = np.zeros(shape=self.num_par_inst, dtype='bool')
        collides_self[~ collides_wall] = self.snake_body_fields[~ collides_wall,
                                                                snake_heads_y[~ collides_wall],
                                                                snake_heads_x[~ collides_wall]]
        dead = collides_wall | collides_self

        collides_fruit = np.all(self.fruit_locations == self.snake_head_locations, axis=1)

        return dead, collides_fruit

    def compute_fields(self):
        snake_heads_y = self.snake_head_locations[~ self.game_overs, 0]
        snake_heads_x = self.snake_head_locations[~ self.game_overs, 1]
        self.snake_body_fields[~ self.game_overs, snake_heads_y, snake_heads_x] = True
        self.snake_head_fields[~ self.game_overs] = False
        self.snake_head_fields[~ self.game_overs, snake_heads_y, snake_heads_x] = True

    def get_states(self):
        fields = np.stack((self.snake_head_fields, self.snake_body_fields, self.fruit_fields), axis=3)
        one_hot_orientations = - np.ones((self.num_par_inst, 4))
        one_hot_orientations[range(self.num_par_inst), self.snake_movement_orientations] = 1
        numeric_state = np.concatenate((one_hot_orientations,
                                        np.expand_dims(self.time_since_last_fruit / MAX_TIME_WITHOUT_SCORE, axis=1)),
                                       axis=1)
        return bool2num(fields), numeric_state

    def get_state_shapes(self):
        image_state_shape = (self.height, self.width, 3)
        numerical_state_shape = 5
        return [image_state_shape, numerical_state_shape]

    def get_config(self):
        config = super(Snake, self).get_config()
        config.update({"height": self.height,
                       "width": self.width})
        return config

    def __init_gui(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        pygame.font.init()
        self.screen = pygame.display.set_mode((100 + self.width * FIELD_SZ, 180 + self.height * FIELD_SZ))

    def render(self):
        if self.screen is None:
            self.__init_gui()

        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                pass

        self.screen.fill(STYLE['bg_color'])

        area_width = FIELD_SZ * self.width + 2
        area_height = FIELD_SZ * self.height + 2

        scr_marg_left = (self.screen.get_width() - area_width) // 2
        scr_marg_top = self.screen.get_height() - area_height - scr_marg_left

        # Grid area
        pygame.draw.rect(self.screen, STYLE['border_color'],
                         [scr_marg_left - 2, scr_marg_top - 2, area_width, area_height])

        # Empty fields
        for y in range(self.height):
            for x in range(self.width):
                pygame.draw.rect(self.screen, STYLE['field_color'],
                                 [scr_marg_left + x * FIELD_SZ, scr_marg_top + y * FIELD_SZ,
                                  FIELD_SZ - 2, FIELD_SZ - 2])

        # Snake
        snake_body_coords = self.snake_bodies.get_body_coords(0)
        length = self.snake_bodies.get_lengths()[0] + 1
        for i, (y, x) in enumerate(snake_body_coords):
            color = np.array(STYLE['snake_color']) * (0.35 * i / length + 0.65) * 0.7
            pygame.draw.rect(self.screen, color,
                             [scr_marg_left + x * FIELD_SZ, scr_marg_top + y * FIELD_SZ, FIELD_SZ - 2, FIELD_SZ - 2])

        # Snake head
        (head_y, head_x) = self.snake_head_locations[0]
        pygame.draw.rect(self.screen, STYLE['snake_color'],
                         [scr_marg_left + head_x * FIELD_SZ, scr_marg_top + head_y * FIELD_SZ, FIELD_SZ - 2,
                          FIELD_SZ - 2])

        # Fruit
        fruit_coords = self.fruit_locations[0]
        pygame.draw.rect(self.screen, STYLE['fruit_color'],
                         [scr_marg_left + fruit_coords[1] * FIELD_SZ,
                          scr_marg_top + fruit_coords[0] * FIELD_SZ, FIELD_SZ - 2, FIELD_SZ - 2])

        # Title text
        font = pygame.font.SysFont('Consolas', 50)
        text = font.render("Snake", True, STYLE["font_color"], None)
        rect = text.get_rect()
        rect.bottomleft = (scr_marg_left, scr_marg_top - 30)
        self.screen.blit(text, rect)

        # Timer text
        font = pygame.font.SysFont('Consolas', 20)
        text = font.render(str(MAX_TIME_WITHOUT_SCORE - self.time_since_last_fruit[0]),
                           True, STYLE["font_color"], None)
        rect = text.get_rect()
        rect.bottomright = (self.screen.get_width() - scr_marg_left - 200, scr_marg_top - 30)
        self.screen.blit(text, rect)

        # Score text
        font = pygame.font.SysFont('Consolas', 30)
        text = font.render(str(self.snake_bodies.get_lengths()[0]), True, STYLE["font_color"], None)
        rect = text.get_rect()
        rect.bottomright = (self.screen.get_width() - scr_marg_left, scr_marg_top - 30)
        self.screen.blit(text, rect)

        pygame.display.flip()

    def state_2d_to_text(self, state):
        state = num2bool(state)
        snake_head_fields = state[:, :, 0]
        snake_body_fields = state[:, :, 1]
        fruit_fields = state[:, :, 2]
        grid_height = snake_head_fields.shape[0]
        grid_width = snake_head_fields.shape[1]

        text = ""

        text += "--" * (grid_width + 2) + "-\n"
        for row in range(grid_height):
            text += " |"
            for col in range(grid_width):
                if snake_head_fields[row, col]:
                    text += " O"
                elif fruit_fields[row, col]:
                    text += " $"
                elif snake_body_fields[row, col]:
                    text += " %"
                else:
                    text += "  "
            text += " |\n"
        text += "--" * (grid_width + 2) + "-"

        return text

    def state_1d_to_text(self, state):
        descriptions = np.array(["up", "right", "down", "left"])
        orientation = descriptions[np.array(state[0:4], dtype='bool')][0]
        text = "- head orientation: %s\n" % orientation + \
               "- time since last fruit: %d" % state[4]
        return text


class SnakeBodyBuffer:
    def __init__(self, length, width):
        self.length = length
        self.width = width

        self.buffer = np.zeros(shape=(self.length, self.width, 2), dtype="int8")
        self.pointer = 0
        self.body_sizes = np.zeros(shape=self.width, dtype="int32")

        self.reset()

    def reset(self):
        self.buffer[:] = 0
        self.pointer = 0
        self.body_sizes[:] = 0

    def reset_for(self, ids):
        self.buffer[:, ids] = 0
        self.body_sizes[ids] = 0

    def init_head_locations(self, ids, locations):
        self.buffer[self.pointer, ids] = locations

    def move_heads(self, new_head_locations):
        self.pointer = (self.pointer + 1) % self.length
        self.buffer[self.pointer] = new_head_locations

    def get_tail_locations(self):
        tail_pointer = self.pointer - self.body_sizes
        return self.buffer[tail_pointer, range(self.width)]

    def grow(self, ids):
        self.body_sizes[ids] += 1

    def get_body_coords(self, idx):
        """Returns the coordinates of the entire snake body for a *single* environment."""
        snake_head_ptr = (self.pointer + 1) % self.length  # Points at buffer entry in front of head entry
        snake_tail_ptr = (self.pointer - self.body_sizes[idx]) % self.length

        if snake_head_ptr <= snake_tail_ptr:
            # Fancy index
            buff_locs = [list(range(snake_tail_ptr, self.length)) + list(range(snake_head_ptr)), idx]
            body_coords = self.buffer[buff_locs]
        else:
            body_coords = self.buffer[snake_tail_ptr:snake_head_ptr, idx]

        return body_coords

    def get_lengths(self):
        return self.body_sizes
