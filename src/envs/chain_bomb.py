from src.envs.env import ParallelEnvironment

import numpy as np
import pygame
import json

OBJ_SYMBOLS = ['T', 'y', 'Y', 'r', 'R', 'o', 'O', 'N']
OBJ_NAMES = ["Targets",
             "Small Yellow Bombs", "Large Yellow Bombs",
             "Small Red Bombs", "Large Red Bombs",
             "Small Orange Bombs", "Large Orange Bombs",
             "Nuggets"]
TARGET_NUMBERS = list(range(1, 9))
TARGET_NUM_DIST = [0.05, 0.1, 0.2, 0.3, 0.2, 0.05, 0.05, 0.05]  # number of targets
NUGGET_NUMBERS = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16]
NUGGET_NUM_DIST = [0.02, 0.05, 0.08, 0.15, 0.2, 0.2, 0.15, 0.08, 0.05, 0.02]  # number of nuggets
INV_BOMB_NUMBERS = list(range(1, 6))
INV_BOMB_NUM_DIST = [0.05, 0.15, 0.5, 0.2, 0.1]  # number of bombs in inventory
FLD_BOMB_NUMBERS = list(range(1, 9))
MAX_NUM_BOMBS = np.max(FLD_BOMB_NUMBERS)
FLD_BOMB_NUM_DIST = [0.02, 0.08, 0.15, 0.25, 0.25, 0.1, 0.1, 0.05]  # number of bombs on field
BOMB_TYPE_LIST = list(range(1, 7))
BOMB_TYPE_DIST = [0.25, 0.1,  # YS YL
                  0.3, 0.15,  # RS RL
                  0.15, 0.05]  # OS OL

BOMB_MAPS = [np.array(((-2, 0),
                       (-1, -1), (-1, 0), (-1, 1),
                       (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
                       (1, -1), (1, 0), (1, 1),
                       (2, 0))),  # YS
             np.array(((-3, 0),
                       (-2, -1), (-2, 0), (-2, 1),
                       (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                       (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
                       (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                       (2, -1), (2, 0), (2, 1),
                       (3, 0))),  # YL
             np.array(((-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 0), (0, 1),
                       (1, -1), (1, 0), (1, 1))),  # RS
             np.array(((-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                       (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                       (-0, -2), (-0, -1), (-0, 0), (-0, 1), (-0, 2),
                       (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                       (2, -2), (2, -1), (2, 0), (2, 1), (2, 2))),  # RL
             np.array(((0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3))),  # OS
             np.array(((-1, -4), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4),
                       (0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                       (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)))  # OL
             ]

POINT_TABLE = [100,  # targets
               20, 40,  # YS YL
               20, 40,  # RS RL
               20, 40,  # OS OL
               40,  # nuggets
               200]  # any bomb left in inventory


class ChainBombBase:
    def __init__(self, num_par_envs=1, height=10, width=20):
        self.num_par_envs = num_par_envs

        self.height = height
        self.width = width

        # Level management
        self.levels_path = "src/envs/cb/levels/%sx%s.json" % (str(self.height), str(self.width))
        self.current_level = 0
        self.levels_list = None

        # State representation
        self.field = np.zeros((1, self.height, self.width, 8), dtype="bool")

        # Environment information
        self.obj_nums = np.zeros((self.num_par_envs, 8), dtype="uint8")
        self.obj_coords = np.zeros((self.num_par_envs, 8, 16, 2), dtype="uint8")
        self.num_bombs_inv = np.zeros(self.num_par_envs, dtype="uint8")
        self.inventories = np.zeros((self.num_par_envs, 5), dtype="uint8")

        self.delete_all_objects(range(self.num_par_envs))

        self.preload_levels_list()

    def delete_all_objects(self, ids):
        self.obj_nums[ids] = 0
        self.obj_coords[ids] = 0
        self.num_bombs_inv[ids] = 0
        self.inventories[ids] = 0

    def preload_levels_list(self):
        try:
            with open(self.levels_path, 'r') as json_file:
                loaded_dict = json.load(json_file)
                dims = loaded_dict["dimensions"]
                if dims[0] != self.height or dims[1] != self.width:
                    raise Exception("Dimensions do not match! The saved levels have dimensions %d x %d but"
                                    "the game/creator accepts only %d x %d." % (dims[0], dims[1],
                                                                                self.height, self.width))
                self.levels_list = loaded_dict["levels"]
        except Exception as e:
            print("No levels loaded.")
            print(e)
            self.levels_list = []
        else:
            print("Loaded levels successfully.")

    def load_level(self, level_no):
        if len(self.levels_list) == 0:
            raise Exception("The levels list is empty! Create levels first before loading levels from the "
                            "levels list.")

        self.delete_all_objects(0)

        print("Loading level %d." % level_no)

        level = self.levels_list[level_no]

        # Get object coordinates
        for obj_type, obj_coords in enumerate(level[0]):
            obj_nums = len(obj_coords)
            self.obj_nums[0, obj_type] = obj_nums
            if obj_nums:
                self.obj_coords[0, obj_type, :obj_nums] = obj_coords

        compute_state_repr(self.field, np.array([0]), self.obj_nums, self.obj_coords)

        # Get inventory
        num_bombs_inv = len(level[1])
        self.inventories[0, :num_bombs_inv] = level[1]
        self.num_bombs_inv[0] = num_bombs_inv

        self.current_level = level_no

    def load_next_level(self):
        next_lvl_no = (self.current_level + 1) % len(self.levels_list)
        self.load_level(next_lvl_no)

    def load_previous_level(self):
        next_lvl_no = (self.current_level - 1) % len(self.levels_list)
        self.load_level(next_lvl_no)

    def reload_level(self):
        self.load_level(self.current_level)

    def user_load_level(self):
        ans = input("Which level do you want to load? Specify a level number:\n")
        try:
            level_no = int(ans)
            self.load_level(level_no)
        except Exception as e:
            print("Invalid input!")
            print(e)


def obj_one_hot_to_idx(obj_one_hot):
    if np.all(obj_one_hot == 0):
        return None
    else:
        obj_type = int(np.argmax(obj_one_hot))
        return obj_type


def obj_one_hot_to_symbol(obj_one_hot, inventory=False):
    if np.all(obj_one_hot == 0):
        return "  "
    else:
        obj_type = int(np.argmax(obj_one_hot))
        if inventory:
            obj_type += 1
        return " " + OBJ_SYMBOLS[obj_type]


def load_highscores_from(path):
    try:
        with open(path, 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(e)
        print("No highscores loaded.")
        return []


def compute_state_repr(fields, ids, obj_nums, obj_coords):
    """Efficiently constructs truth tables encoding object locations on the field."""
    fields[ids] = 0
    # Iterate over object "slots" in coordinate matrix
    for obj_idx in range(obj_coords.shape[2]):
        obj_existing = obj_idx < obj_nums[ids]
        if not np.any(obj_existing):
            break
        for obj_type in range(8):
            envs_with_ex_obj = ids[obj_existing[:, obj_type]]
            fields[envs_with_ex_obj,
                   obj_coords[envs_with_ex_obj, obj_type, obj_idx, 0],
                   obj_coords[envs_with_ex_obj, obj_type, obj_idx, 1],
                   obj_type] = True


class ChainBomb(ParallelEnvironment, ChainBombBase):
    NAME = "chain_bomb"
    LEVELS = True
    TIME_RELEVANT = False
    WINS_RELEVANT = True

    # Level selection mode
    TRAIN_MODE = 0  # purely random level generation
    TEST_MODE = 1  # ascending level selection from levels.json file

    # Player mode
    AI = 0
    HUMAN = 1

    # Highscore management
    HIGHSCORE_PATH_AI = "src/envs/cb/highscores_ai.json"
    HIGHSCORE_PATH_HUMAN = "src/envs/cb/highscores_human.json"

    def __init__(self, num_par_envs, height=10, width=20):
        actions = get_actions(height, width)
        ParallelEnvironment.__init__(self, num_par_envs, actions)
        ChainBombBase.__init__(self, num_par_envs, height, width)

        # Level management
        self.mode = self.TRAIN_MODE  # level selection mode
        self.player = self.AI

        # State representation
        self.fields = np.zeros((self.num_par_envs, self.height, self.width, 8), dtype="bool")

        # Highscore management
        self.highscores_ai = None
        self.highscores_human = None
        self.load_highscores()

        # Auxiliary for step calculation
        self.explosions_active = np.zeros(self.num_par_envs, dtype="bool")
        self.explosions_fields = np.zeros((self.num_par_envs, self.height, self.width, 1), dtype="bool")

        # GUI rendering
        self.renderer = CBGameRenderer(self.height, self.width)
        self.explosions_list = []

        self.reset()

    def reset(self, lvl_no=None):
        self.reset_for(np.arange(self.num_par_envs), lvl_no)

    def reset_for(self, ids, lvl_no=None):
        self.scores[ids] = 0
        self.times[ids] = 0
        self.game_overs[ids] = False
        self.wins[ids] = False

        if self.mode == self.TRAIN_MODE:
            self.delete_all_objects(ids)
            self.generate_random_levels(ids)
        elif self.mode == self.TEST_MODE:
            if lvl_no is not None:
                self.load_level(lvl_no)
            else:
                self.reload_level()

        compute_state_repr(self.fields, ids, self.obj_nums, self.obj_coords)

    def reset_to_next_level(self):
        next_lvl_no = (self.current_level + 1) % len(self.levels_list)
        self.reset(next_lvl_no)

    def reset_to_previous_level(self):
        next_lvl_no = (self.current_level - 1) % len(self.levels_list)
        self.reset(next_lvl_no)

    def step(self, actions):
        points = self.ignite_explosion(actions)

        # Use up bomb
        self.inventories[range(self.num_par_envs), self.times] = 0
        self.num_bombs_inv -= 1

        self.times += 1

        # End game if all bombs are used up
        self.game_overs[:] = self.num_bombs_inv == 0

        # Declare win if all targets are eliminated
        all_targets_eliminated = self.obj_nums[:, 0] == 0
        self.game_overs[all_targets_eliminated] = True
        self.wins[all_targets_eliminated] = True

        # Sum up points of all bombs left in inventory
        points[self.wins] += self.num_bombs_inv[self.wins].astype("int32") * POINT_TABLE[8]

        assert np.all(points % 10 == 0)

        self.scores += points
        reward = points / 100

        # Nullify score for lost levels
        lost = self.game_overs & ~ self.wins
        self.scores[lost] = 0
        reward[lost] = -3

        if self.mode == self.TEST_MODE and self.game_overs[0] and self.wins[0]:
            self.update_highscore()

        return reward, self.scores, self.game_overs, self.times, self.wins

    def generate_random_levels(self, ids):
        n = len(ids)

        # Generate random numbers of targets, nuggets, bombs in inventory, and bombs on field
        num_targets = np.random.choice(TARGET_NUMBERS, n, p=TARGET_NUM_DIST)
        num_nuggets = np.random.choice(NUGGET_NUMBERS, n, p=NUGGET_NUM_DIST)
        num_bombs_inv = np.random.choice(INV_BOMB_NUMBERS, n, p=INV_BOMB_NUM_DIST)
        num_bombs_field = np.random.choice(FLD_BOMB_NUMBERS, n, p=FLD_BOMB_NUM_DIST)

        # Assign random bomb types
        bomb_types_inv = np.random.choice(BOMB_TYPE_LIST, (n, 5), p=BOMB_TYPE_DIST)
        bomb_types_field = np.random.choice(BOMB_TYPE_LIST, (n, MAX_NUM_BOMBS), p=BOMB_TYPE_DIST)

        # Throw away abundant bombs
        for i in range(n):
            bomb_types_field[i, num_bombs_field[i]:] = 0

        # Save object numbers
        self.obj_nums[ids, 0] = num_targets
        for bomb_type in BOMB_TYPE_LIST:
            type_counts = np.sum(bomb_types_field == bomb_type, axis=1)
            self.obj_nums[ids, bomb_type] = type_counts
        self.obj_nums[ids, 7] = num_nuggets

        # Put bombs into inventories
        self.inventories[ids] = bomb_types_inv
        self.num_bombs_inv[ids] = num_bombs_inv

        self.obj_coords[ids] = self.generate_random_coords(n)

    def generate_random_coords(self, num_fields):
        """Generates random object coordinates"""
        locs = np.tile(np.array([range(self.height * self.width)]), (num_fields, 1))
        shuffle_row_wise(locs)
        selected_rand_locs = locs[:, :8 * 16].reshape((num_fields, 8, 16))
        return self.locs_to_coords(selected_rand_locs)

    def ignite_explosion(self, actions):
        dropped_bombs = self.inventories[range(self.num_par_envs), self.times] - 1
        drop_coords = self.locs_to_coords(actions)

        points = np.zeros(self.num_par_envs, dtype='int32')

        # Calculate initial explosions fields
        ignited_bombs_locations = np.stack((np.arange(self.num_par_envs),
                                            drop_coords[:, 0],
                                            drop_coords[:, 1],
                                            dropped_bombs), axis=1)
        self.embed_explosions(ignited_bombs_locations)
        self.explosions_active[:] = True

        # Compute chain explosions iteratively
        while np.any(self.explosions_active):
            # Add up points of all hit objects
            hit_objects = self.fields[self.explosions_active] & self.explosions_fields[self.explosions_active]
            sum_hit_objects = np.sum(hit_objects, axis=(1, 2), dtype="uint8")
            points[self.explosions_active] += np.dot(sum_hit_objects, POINT_TABLE[:-1])

            # Identify all bombs hit by the explosions
            hit_bombs = hit_objects[:, :, :, 1:-1]
            hit_bombs_locations = np.stack(np.where(hit_bombs), axis=1)
            envs_with_explosions = np.arange(self.num_par_envs)[self.explosions_active]
            hit_bombs_locations[:, 0] = envs_with_explosions[hit_bombs_locations[:, 0]]

            # Eliminate all objects inside explosion areas
            self.fields[self.explosions_active] &= ~ self.explosions_fields[self.explosions_active]
            self.obj_nums[self.explosions_active] -= sum_hit_objects

            # Let all the bombs hit by the explosions also explode
            self.embed_explosions(hit_bombs_locations)

            # Identify envs where the explosions came to an end
            sum_hit_bombs = sum_hit_objects[:, 1:-1]
            bomb_hit = np.sum(sum_hit_bombs, axis=1) > 0
            self.explosions_active[self.explosions_active] = bomb_hit

        # Reset explosions field
        self.explosions_fields[:] = 0
        self.explosions_list = []

        return points

    def embed_explosions(self, exploding_bombs):
        """Takes a 2D array where rows represent exploding bombs. Each row has 4 entries:
        [env_id, explosion_center_y, explosion_center_x, exploding_bomb_type]."""
        for env_id, explosion_center_y, explosion_center_x, exploding_bomb_type in exploding_bombs:
            explosion_center = (explosion_center_y, explosion_center_x)
            explosion_area = BOMB_MAPS[exploding_bomb_type] + explosion_center

            if self.renderer.gui_initialized:
                self.explosions_list += [(exploding_bomb_type + 1, explosion_center)]
                self.render()
                pygame.time.wait(350)

            # Cut off overhang
            overhang = np.zeros(len(explosion_area), dtype="bool")
            overhang[(explosion_area[:, 0] < 0) | (explosion_area[:, 0] >= self.height)] = True
            overhang[(explosion_area[:, 1] < 0) | (explosion_area[:, 1] >= self.width)] = True

            explosion_area = explosion_area[~ overhang]

            area_y = explosion_area[:, 0]
            area_x = explosion_area[:, 1]

            self.explosions_fields[env_id, area_y, area_x] = True

    def check_for_object_at(self, position_coords, env_id):
        for obj_type in range(8):
            for obj_idx, obj_coords in enumerate(self.obj_coords[env_id, obj_type, 0:self.obj_nums[env_id, obj_type]]):
                if np.all(obj_coords == position_coords):
                    return obj_type, obj_idx
        return None, None

    def delete_object(self, obj_type, obj_idx, env_id):
        if obj_idx + 1 != self.obj_nums[env_id, obj_type]:
            self.obj_coords[env_id, obj_type, obj_idx:self.obj_nums[env_id, obj_type] - 1] = \
                self.obj_coords[env_id, obj_type, obj_idx + 1:self.obj_nums[env_id, obj_type]]
        self.obj_nums[env_id, obj_type] -= 1

    def set_mode(self, mode):
        """Sets the level selection/generation mode."""
        if mode == self.TRAIN_MODE:
            self.current_level = None
            self.mode = mode
            self.reset()
        elif mode == self.TEST_MODE:
            if self.num_par_envs == 1:
                self.current_level = 0
                self.mode = mode
                self.reset()
            else:
                print("Warning: Test mode only allowed if environment simulates a single instance. "
                      "This environment, however, has %d parallel instances." % self.num_par_envs)
        else:
            raise Exception("Invalid mode number given.")

    def load_highscores(self):
        num_levels = len(self.levels_list)

        self.highscores_ai = np.zeros(num_levels)
        loaded_highscores_ai = load_highscores_from(self.HIGHSCORE_PATH_AI)
        self.highscores_ai[:len(loaded_highscores_ai)] = loaded_highscores_ai

        self.highscores_human = np.zeros(num_levels)
        loaded_highscores_human = load_highscores_from(self.HIGHSCORE_PATH_HUMAN)
        self.highscores_human[:len(loaded_highscores_human)] = loaded_highscores_human

    def save_highscores(self):
        if self.player == self.AI:
            with open(self.HIGHSCORE_PATH_AI, 'w') as json_file:
                json.dump(list(self.highscores_ai), json_file)
        else:
            with open(self.HIGHSCORE_PATH_HUMAN, 'w') as json_file:
                json.dump(list(self.highscores_human), json_file)

    def update_highscore(self):
        if self.player == self.AI:
            self.highscores_ai[self.current_level] = \
                np.max((self.highscores_ai[self.current_level], self.scores))
        else:
            self.highscores_human[self.current_level] = \
                np.max((self.highscores_human[self.current_level], self.scores))
        self.save_highscores()

    def get_highscores(self):
        return self.highscores_ai, self.highscores_human

    def get_states(self):
        # Create one-hot representation for each inventory slot
        inventory_one_hot = np.zeros((self.num_par_envs, 5, 6), dtype="bool")
        for slot in range(5):
            bomb_in_slot = slot < self.num_bombs_inv
            current_slot = (self.times + slot)[bomb_in_slot]
            bomb_types_in_slot = self.inventories[bomb_in_slot, current_slot]
            inventory_one_hot[bomb_in_slot, slot, bomb_types_in_slot - 1] = 1

        return np.copy(self.fields), inventory_one_hot.reshape((self.num_par_envs, -1))

    def get_state_shapes(self):
        image_state_shape = [self.height, self.width, 8]
        numerical_state_shape = 5 * 6
        return image_state_shape, numerical_state_shape

    def image_state_to_text(self, state):
        grid_height = state.shape[0]
        grid_width = state.shape[1]

        text = ""
        text += "--" * (grid_width + 2) + "-\n"
        for row in range(grid_height):
            text += " |"
            for col in range(grid_width):
                obj_one_hot = state[row, col]
                text += obj_one_hot_to_symbol(obj_one_hot)
            text += " |\n"
        text += "--" * (grid_width + 2) + "-"
        return text

    def numerical_state_to_text(self, numerical_state):
        text = "Inventory:"
        for obj_one_hot in np.reshape(numerical_state, (5, 6)):
            text += obj_one_hot_to_symbol(obj_one_hot, True)
        return text

    def locs_to_coords(self, locs):
        """Converts scalar locations into 2D coordinates. Also works with actions as action ID == location ID."""
        return np.stack(np.unravel_index(locs, (self.height, self.width)), axis=-1)

    def coords_to_action(self, coords):
        action = coords[0] * self.width + coords[1]
        return action

    def has_test_levels(self):
        return len(self.levels_list) > 0

    def generate_pretrain_data(self, num_instances):
        self.generate_random_levels(range(num_instances))
        return self.get_states()[0]

    def run_for_human(self):
        self.player = self.HUMAN
        self.load_highscores()
        score_history = []
        levels_played = 0

        self.renderer.init_gui()

        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                # Did the user click the window close button?
                if event.type == pygame.QUIT:
                    running = False  # not return?
                elif event.type == pygame.MOUSEMOTION:
                    pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # mousewheel up
                        self.reset_to_previous_level()
                    elif event.button == 5:  # mousewheel down
                        self.reset_to_next_level()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # right mouseclick
                        self.reset()
                    elif event.button == 1:  # left mouseclick
                        if not self.game_overs[0]:
                            field_coords = self.renderer.mouse_pos_to_field_coords(mouse_pos)
                            if field_coords_valid(field_coords, self.height, self.width):
                                action = self.coords_to_action(field_coords)
                                self.step([action])
                        else:
                            if self.mode == self.TEST_MODE:
                                if self.wins[0]:
                                    self.reset_to_next_level()
                                else:
                                    self.reset()
                            else:
                                score_history += [self.scores[0]]
                                levels_played += 1
                                print("Your statistics -- average score: %.0f -- %d levels played" %
                                      (np.average(score_history), levels_played))
                                self.reset()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_l:  # keyboard "l"
                        self.user_load_level()

            if self.game_overs[0]:
                mouse_pos = None
            self.render(mouse_pos)

        pygame.quit()

    def render(self, mouse_pos=None):
        highscores_ai = self.highscores_ai[self.current_level] if self.mode == self.TEST_MODE else None
        highscores_human = self.highscores_human[self.current_level] if self.mode == self.TEST_MODE else None

        self.renderer.render_game(self.fields[0], self.inventories[0], self.num_bombs_inv[0],
                                  self.times[0], self.scores[0], highscores_ai, highscores_human,
                                  mouse_pos, self.game_overs[0], self.wins[0], self.explosions_list)


class CBCreator(ChainBombBase):
    """Used to create new ChainBomb levels with a GUI."""

    def __init__(self, height=10, width=20):
        ChainBombBase.__init__(self, 1, height, width)

        self.renderer = CBCreatorRenderer(self.height, self.width)
        self.drag = False
        self.dragged_obj = None

        self.new_level()

        self.run_creator()

    def new_level(self):
        self.field[:] = 0
        self.obj_nums[:] = 0
        self.obj_coords[:] = 0
        self.inventories[:] = 0
        self.current_level = len(self.levels_list)
        print("Creating new level: Level %d." % self.current_level)

    def save_levels(self):
        try:
            with open(self.levels_path, 'w') as json_file:
                save_dict = {"dimensions": (self.height, self.width),
                             "levels": self.levels_list}
                json.dump(save_dict, json_file)
        except Exception as e:
            print(e)
            print("No levels saved.")

    def save_current_level(self):
        # Check for completeness
        if np.sum(self.inventories[0]) == 0:
            raise Exception("Inventory is empty! At least one bomb must be in the inventory. "
                            "Or, how do you play levels without bombs?")
        if self.obj_nums[0, 0] == 0:
            raise Exception("Targets missing! At least one target must be on the field. "
                            "Give the game a meaning.")

        # Check for correctness
        for obj_type, obj_num in enumerate(self.obj_nums[0]):
            if obj_num > 16:
                raise Exception("Too many objects! You have %d %s on the field "
                                "but no more than 16 per object type are allowed." % (obj_num, OBJ_NAMES[obj_type]))

        # Reformat level
        inv = []
        for bomb in self.inventories[0]:
            if bomb != 0:
                inv += [int(bomb)]

        obj_coords = []
        for obj_type, objs in enumerate(self.obj_coords[0]):
            objs = []
            for i in range(self.obj_nums[0, obj_type]):
                coords = self.obj_coords[0, obj_type, i]
                objs += [[int(coords[0]), int(coords[1])]]
            obj_coords += [objs]

        # Add level to list
        if self.current_level >= len(self.levels_list):
            self.levels_list += [[obj_coords, inv]]
            self.save_levels()
            print("Level added to level list as Level %d and saved." % (len(self.levels_list) - 1))
        else:
            self.levels_list[self.current_level] = [obj_coords, inv]
            self.save_levels()
            print("Saved Level %d." % int(self.current_level))

    def drop_obj_to_field(self, obj_type, field_coords):
        if self.obj_nums[0, obj_type] + 1 > 16:
            raise Exception("There are already 16 objects of this kind and no more are allowed.")

        # Delete object if it is in the way
        target_obj_type, target_obj_idx = self.get_object_at(field_coords)
        if target_obj_type is not None:
            self.del_object(target_obj_type, target_obj_idx)

        # Add new object
        self.obj_coords[0, obj_type, self.obj_nums[0, obj_type]] = field_coords
        self.obj_nums[0, obj_type] += 1

        if self.obj_nums[0, obj_type] > 16:
            print("Warning: max. 16 objects per type are allowed!")

        compute_state_repr(self.field, np.array([0]), self.obj_nums, self.obj_coords)

    def drop_obj_to_inventory(self, obj_type, inv_field):
        if obj_type not in [1, 2, 3, 4, 5, 6]:
            raise Exception("Only bombs are allowed in inventory.")

        self.inventories[0, inv_field] = self.dragged_obj

    def del_object(self, obj_type, obj_idx):
        if obj_idx + 1 != self.obj_nums[0, obj_type]:
            self.obj_coords[0, obj_type, obj_idx:self.obj_nums[0, obj_type] - 1] = \
                self.obj_coords[0, obj_type, obj_idx + 1:self.obj_nums[0, obj_type]]
        self.obj_nums[0, obj_type] -= 1

        compute_state_repr(self.field, np.array([0]), self.obj_nums, self.obj_coords)

    def get_object_at(self, field_coords):
        for obj_type, obj_coords in enumerate(self.obj_coords[0]):
            for obj_idx, occupied_position in enumerate(obj_coords[:self.obj_nums[0, obj_type]]):
                if np.all(occupied_position == field_coords):
                    return obj_type, obj_idx
        return None, None

    def mouse_pos_to_field_obj(self, mouse_pos):
        field_coords = self.renderer.mouse_pos_to_field_coords(mouse_pos)
        obj_type, obj_idx = self.get_object_at(field_coords)
        if obj_type is not None:
            self.del_object(obj_type, obj_idx)
        return obj_type

    def run_creator(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():

                if event.type == pygame.QUIT:  # window close button
                    running = False  # not return?

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # left mouseclick
                        toolbox_obj = self.renderer.mouse_pos_to_toolbox_obj(mouse_pos)
                        field_obj = self.mouse_pos_to_field_obj(mouse_pos)
                        inv_field = self.renderer.mouse_pos_to_inv_field(mouse_pos)

                        if toolbox_obj is not None:
                            self.drag = True
                            self.dragged_obj = toolbox_obj
                        elif field_obj is not None:
                            self.drag = True
                            self.dragged_obj = field_obj
                        elif inv_field is not None:
                            if self.inventories[0, inv_field] != 0:
                                self.drag = True
                                self.dragged_obj = self.inventories[0, inv_field]
                                self.inventories[0, inv_field] = 0
                    if event.button == 4:  # mousewheel up
                        self.load_previous_level()
                    elif event.button == 5:  # mousewheel down
                        self.load_next_level()

                elif event.type == pygame.MOUSEMOTION:
                    pass

                elif event.type == pygame.MOUSEBUTTONUP:
                    field_coords = self.renderer.mouse_pos_to_field_coords(mouse_pos)
                    inv_field = self.renderer.mouse_pos_to_inv_field(mouse_pos)
                    if event.button == 1:  # left mouseclick
                        if field_coords_valid(field_coords, self.height, self.width) and self.drag:
                            try:
                                self.drop_obj_to_field(self.dragged_obj, field_coords)
                            except Exception as e:
                                print(e)
                        elif inv_field is not None:
                            try:
                                self.drop_obj_to_inventory(self.dragged_obj, inv_field)
                            except Exception as e:
                                print(e)
                        self.drag = False
                        self.dragged_obj = None
                        pygame.mouse.set_cursor(*pygame.cursors.arrow)
                    elif event.button == 3:  # right mouseclick
                        # Delete object
                        if field_coords_valid(field_coords, self.height, self.width):
                            obj_type, obj_idx = self.get_object_at(field_coords)
                            if obj_type is not None:
                                self.del_object(obj_type, obj_idx)
                        elif inv_field is not None:
                            self.inventories[0, inv_field] = 0

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:  # keyboard "n"
                        self.new_level()
                    elif event.key == pygame.K_s:  # keyboard "a"
                        try:
                            self.save_current_level()
                        except Exception as e:
                            print(e)
                    elif event.key == pygame.K_l:  # keyboard "l"
                        self.user_load_level()

            self.render(mouse_pos)

    def render(self, mouse_pos):
        self.renderer.render_creator(self.field[0], self.inventories[0], mouse_pos,
                                     self.drag, dragged_obj=self.dragged_obj)


class CBBaseRenderer:
    SQUARE_SZ = 30
    OBJ_COLOR_NAMES = ["targets",
                       "bomb_type1", "bomb_type1",
                       "bomb_type2", "bomb_type2",
                       "bomb_type3", "bomb_type3",
                       "nuggets"]

    COLORS = {"background": (50, 50, 50),
              "field": (70, 70, 70),
              "border": (130, 130, 130),
              "font": (200, 200, 200),
              "targets": (50, 200, 255),
              "bomb_type1": (255, 255, 10),
              "bomb_type2": (240, 30, 30),
              "bomb_type3": (240, 150, 30),
              "nuggets": (180, 180, 180),
              "inv_highlight": (120, 120, 120),
              "hover": (255, 255, 255, 200)}

    def __init__(self, field_height, field_width):
        self.f_height = field_height
        self.f_width = field_width

        self.field_width = self.SQUARE_SZ * self.f_width + 2
        self.field_height = self.SQUARE_SZ * self.f_height + 2

        self.field_marg_left = None
        self.field_marg_top = None

        self.inventory_anchor = None

        self.screen = None
        self.gui_initialized = False

    def init_gui(self):
        pygame.init()
        pygame.display.set_caption("ChainBomb")
        pygame.font.init()
        self.screen = pygame.display.set_mode((100 + self.f_width * self.SQUARE_SZ,
                                               180 + self.f_height * self.SQUARE_SZ))

        self.field_marg_left = (self.screen.get_width() - self.field_width) // 2
        self.field_marg_top = self.screen.get_height() - self.field_height - self.field_marg_left

        self.inventory_anchor = (self.screen.get_width() - self.field_marg_left - 200, 40)

        self.gui_initialized = True

    def render_field(self, field):
        # Draw base box (used for borders)
        pygame.draw.rect(self.screen, self.COLORS['border'],
                         [self.field_marg_left - 2, self.field_marg_top - 2, self.field_width, self.field_height])

        for y, row in enumerate(field):
            for x, one_hot_cell in enumerate(row):
                obj_type = obj_one_hot_to_idx(one_hot_cell)
                if obj_type is not None:
                    box_bounds = self.coords_to_box_boundaries((y, x))
                    pygame.draw.rect(self.screen, self.COLORS[self.OBJ_COLOR_NAMES[obj_type]], box_bounds)
                    if obj_type in [1, 3, 5]:
                        inner_box_bounds = [box_bounds[0] + 2, box_bounds[1] + 2,
                                            box_bounds[2] - 4, box_bounds[3] - 4]
                        pygame.draw.rect(self.screen, self.COLORS['field'], inner_box_bounds)
                else:
                    pygame.draw.rect(self.screen, self.COLORS['field'],
                                     self.coords_to_box_boundaries([y, x]))

    def render_hover_explosion_shape(self, mouse_pos, obj_type):
        drop_coords = self.mouse_pos_to_field_coords(mouse_pos)

        if field_coords_valid(drop_coords, self.f_height, self.f_width):
            hover_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
            hover_surface.set_alpha(60)
            hover_surface.set_colorkey((0, 0, 0))

            if obj_type in [1, 2, 3, 4, 5, 6]:
                explosion_shape = BOMB_MAPS[obj_type - 1]
                explosion_area = explosion_shape + drop_coords
                for box in explosion_area:
                    if field_coords_valid(box, self.f_height, self.f_width):
                        box_bounds = self.coords_to_box_boundaries([box[0], box[1]])
                        pygame.draw.rect(hover_surface, (255, 255, 255), box_bounds)
            else:
                box_bounds = self.coords_to_box_boundaries(drop_coords)
                pygame.draw.rect(hover_surface, (255, 255, 255), box_bounds)

            pygame.mouse.set_cursor(*pygame.cursors.broken_x)
            self.screen.blit(hover_surface, (0, 0))
        else:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)

    def render_explosions(self, explosions_list):
        for bomb_type, bomb_pos in explosions_list:
            explosion_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
            explosion_surface.set_alpha(160)
            explosion_surface.set_colorkey((0, 0, 0))

            explosion_shape = BOMB_MAPS[bomb_type - 1]
            explosion_area = explosion_shape + bomb_pos

            for box in explosion_area:
                if field_coords_valid(box, self.f_height, self.f_width):
                    box_bounds = self.coords_to_box_boundaries([box[0], box[1]])
                    explosion_color = self.OBJ_COLOR_NAMES[bomb_type]
                    pygame.draw.rect(explosion_surface, self.COLORS[explosion_color], box_bounds)

            self.screen.blit(explosion_surface, (0, 0))

    def render_inventory(self, inventory, num_inv_bombs, turn_no, game_over):
        for i, bomb in enumerate(inventory):
            box_bounds = [self.inventory_anchor[0] + i * 40,
                          self.inventory_anchor[1],
                          24, 24]
            inner_box_bounds = [box_bounds[0] + 2, box_bounds[1] + 2,
                                box_bounds[2] - 4, box_bounds[3] - 4]

            # Current bomb highlight
            if turn_no is not None and not game_over and i == turn_no:
                outer_box_bounds = [box_bounds[0] - 8, box_bounds[1] - 8,
                                    box_bounds[2] + 16, box_bounds[3] + 16]
                pygame.draw.rect(self.screen, self.COLORS['inv_highlight'], outer_box_bounds)
                inner_color = self.COLORS['inv_highlight']
            else:
                inner_color = self.COLORS['background']

            if bomb == 0 or (turn_no is not None and i >= num_inv_bombs + turn_no):
                # Draw empty inv field
                pygame.draw.rect(self.screen, self.COLORS['field'], box_bounds)
                pygame.draw.rect(self.screen, inner_color, inner_box_bounds)
                line_start = [box_bounds[0] - 4, box_bounds[1] + box_bounds[2] + 3]
                line_end = [box_bounds[0] + box_bounds[2] + 3, box_bounds[1] - 4]
                pygame.draw.line(self.screen, self.COLORS['field'], line_start, line_end, 3)
            else:
                # Draw bomb
                pygame.draw.rect(self.screen, self.COLORS[self.OBJ_COLOR_NAMES[bomb]], box_bounds)
                if bomb in [1, 3, 5]:
                    pygame.draw.rect(self.screen, inner_color, inner_box_bounds)

    def render_score(self, score):
        font = pygame.font.SysFont('Consolas', 30)
        text = font.render(str(score), True, self.COLORS["font"], None)
        rect = text.get_rect()
        rect.topleft = (self.inventory_anchor[0], self.inventory_anchor[1] + 40)
        self.screen.blit(text, rect)

    def render_title(self):
        font = pygame.font.SysFont('Consolas', 50)
        text = font.render("ChainBomb", True, self.COLORS["font"], None)
        rect = text.get_rect()
        rect.bottomleft = (self.field_marg_left, self.field_marg_top - 30)
        self.screen.blit(text, rect)

    def render_game_over_screen(self, won, score, highscore_ai, highscore_human):
        # Dimmer
        dimmer = pygame.Surface((self.field_width, self.field_height))
        dimmer.set_alpha(220)
        dimmer.fill(self.COLORS["background"])
        self.screen.blit(dimmer, (self.field_marg_left - 2, self.field_marg_top - 2))

        # Game Over text
        font = pygame.font.SysFont('Consolas', 50)
        if won:
            game_over_text = "You Won!"
        else:
            game_over_text = "You Lost..."
            score = 0
        text = font.render(game_over_text, True, self.COLORS["font"], None)
        rect = text.get_rect()
        rect.center = (self.field_marg_left + self.field_width // 2,
                       self.field_marg_top + self.field_height // 2 - 60)
        self.screen.blit(text, rect)

        # Scores
        font = pygame.font.SysFont('Consolas', 30)

        # Score
        text = font.render("Score:", True, self.COLORS["font"], None)
        rect = text.get_rect()
        rect.midright = (self.field_marg_left + self.field_width // 2,
                         self.field_marg_top + self.field_height // 2 + 20)
        self.screen.blit(text, rect)
        text = font.render(" %d" % score, True, self.COLORS["font"], None)
        rect = text.get_rect()
        rect.midleft = (self.field_marg_left + self.field_width // 2,
                        self.field_marg_top + self.field_height // 2 + 20)
        self.screen.blit(text, rect)

        # AI highscore
        if highscore_ai is not None:
            text = font.render("AI Highscore:", True, self.COLORS["font"], None)
            rect = text.get_rect()
            rect.midright = (self.field_marg_left + self.field_width // 2,
                             self.field_marg_top + self.field_height // 2 + 55)
            self.screen.blit(text, rect)
            text = font.render(" %d" % highscore_ai, True, self.COLORS["font"], None)
            rect = text.get_rect()
            rect.midleft = (self.field_marg_left + self.field_width // 2,
                            self.field_marg_top + self.field_height // 2 + 55)
            self.screen.blit(text, rect)

        # Human highscore
        if highscore_human is not None:
            text = font.render("Human Highscore:", True, self.COLORS["font"], None)
            rect = text.get_rect()
            rect.midright = (self.field_marg_left + self.field_width // 2,
                             self.field_marg_top + self.field_height // 2 + 90)
            self.screen.blit(text, rect)
            text = font.render(" %d" % highscore_human, True, self.COLORS["font"], None)
            rect = text.get_rect()
            rect.midleft = (self.field_marg_left + self.field_width // 2,
                            self.field_marg_top + self.field_height // 2 + 90)
            self.screen.blit(text, rect)

    def coords_to_box_boundaries(self, coords):
        return [self.field_marg_left + coords[1] * self.SQUARE_SZ,
                self.field_marg_top + coords[0] * self.SQUARE_SZ,
                self.SQUARE_SZ - 2, self.SQUARE_SZ - 2]

    def mouse_pos_to_field_coords(self, mouse_pos):
        field_y = mouse_pos[1] - self.field_marg_top
        field_x = mouse_pos[0] - self.field_marg_left
        y_coord = field_y // self.SQUARE_SZ
        x_coord = field_x // self.SQUARE_SZ
        return y_coord, x_coord


class CBGameRenderer(CBBaseRenderer):
    """Renders the GUI for ChainBomb."""

    def __init__(self, field_height, field_width):
        super(CBGameRenderer, self).__init__(field_height, field_width)

    def render_game(self, field, inventory, num_inv_bombs, turn_no, score, highscore_ai=None, highscore_human=None,
                    mouse_pos=None, game_over=False, won=False, explosions_list=None):
        if not self.gui_initialized:
            self.init_gui()

        self.screen.fill(self.COLORS['background'])
        self.render_field(field)

        if mouse_pos:
            self.render_hover_explosion_shape(mouse_pos, inventory[turn_no])

        if explosions_list:
            self.render_explosions(explosions_list)

        self.render_inventory(inventory, num_inv_bombs, turn_no, game_over)
        self.render_score(score)
        self.render_title()

        if game_over:
            self.render_game_over_screen(won, score, highscore_ai, highscore_human)

        pygame.display.flip()


class CBCreatorRenderer(CBBaseRenderer):
    """Renders the GUI for CBCreator."""

    def __init__(self, field_height, field_width):
        super(CBCreatorRenderer, self).__init__(field_height, field_width)
        self.init_gui()

    def render_creator(self, field, inventory, mouse_pos=None, drag=False, dragged_obj=None):
        self.screen.fill(self.COLORS['background'])
        self.render_field(field)

        self.render_toolbox(mouse_pos)
        self.render_inventory(inventory, 5, self.mouse_pos_to_inv_field(mouse_pos), False)

        if mouse_pos and drag:
            self.render_dragged_obj(mouse_pos, dragged_obj)
            self.render_hover_explosion_shape(mouse_pos, dragged_obj)

        pygame.display.flip()

    def render_toolbox(self, mouse_pos):
        toolbox_anchor = [self.field_marg_left, 40]
        hovered_obj = self.mouse_pos_to_toolbox_obj(mouse_pos)
        for obj_type, obj_color_name in enumerate(self.OBJ_COLOR_NAMES):
            box_bounds = [toolbox_anchor[0] + obj_type * 40,
                          toolbox_anchor[1],
                          24, 24]
            inner_box_bounds = [box_bounds[0] + 2, box_bounds[1] + 2,
                                box_bounds[2] - 4, box_bounds[3] - 4]

            inner_color = self.COLORS['background']

            if obj_type == hovered_obj:
                outer_box_bounds = [box_bounds[0] - 8, box_bounds[1] - 8,
                                    box_bounds[2] + 16, box_bounds[3] + 16]
                pygame.draw.rect(self.screen, self.COLORS['inv_highlight'], outer_box_bounds)
                inner_color = self.COLORS['inv_highlight']

            pygame.draw.rect(self.screen, self.COLORS[self.OBJ_COLOR_NAMES[obj_type]], box_bounds)
            if obj_type in [1, 3, 5]:
                pygame.draw.rect(self.screen, inner_color, inner_box_bounds)

    def render_dragged_obj(self, mouse_pos, obj_type):
        box_bounds = [mouse_pos[0] - 12, mouse_pos[1] - 12, 24, 24]
        pygame.draw.rect(self.screen, self.COLORS[self.OBJ_COLOR_NAMES[obj_type]], box_bounds)
        if obj_type in [1, 3, 5]:
            inner_box_bounds = [box_bounds[0] + 2, box_bounds[1] + 2,
                                box_bounds[2] - 4, box_bounds[3] - 4]
            pygame.draw.rect(self.screen, self.COLORS['background'], inner_box_bounds)

    def mouse_pos_to_toolbox_obj(self, mouse_pos):
        y = mouse_pos[1] - 40 + 8
        x = mouse_pos[0] - self.field_marg_left + 8
        obj_type = x // 40
        if 0 <= y < 40 and 0 <= obj_type < 8:
            return obj_type
        else:
            return None

    def mouse_pos_to_inv_field(self, mouse_pos):
        y = mouse_pos[1] - self.inventory_anchor[1] + 8
        x = mouse_pos[0] - self.inventory_anchor[0] + 8
        inv_field = x // 40
        if 0 <= y < 40 and 0 <= inv_field < 5:
            return inv_field
        else:
            return None


def field_coords_valid(field_coords, field_height, field_width):
    return 0 <= field_coords[0] < field_height and 0 <= field_coords[1] < field_width


def get_actions(height, width):
    actions = []
    for row in range(height):
        for col in range(width):
            actions += ["row %d, col %d" % (row, col)]
    return actions


def shuffle_row_wise(matrix):
    """Shuffles not efficiently along axis=1 in-place."""
    # TODO: make more efficient
    for row in matrix:
        np.random.shuffle(row)
