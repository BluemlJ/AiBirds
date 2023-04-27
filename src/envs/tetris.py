import pygame
import numpy as np
from typing import Sequence

from src.envs.env import ParallelEnvironment
from src.utils.math import sigmoid, get_2d_rotation_matrix

FIGURES = np.array([
    [[2, 0], [2, 1], [2, 2], [2, 3]],  # I
    [[0, 2], [1, 2], [2, 2], [2, 1]],  # J
    [[0, 1], [1, 1], [2, 1], [2, 2]],  # L
    [[1, 1], [1, 2], [2, 1], [2, 2]],  # O
    [[1, 2], [2, 2], [2, 1], [3, 1]],  # S
    [[2, 1], [2, 2], [2, 3], [3, 2]],  # T
    [[2, 1], [2, 2], [1, 2], [1, 3]],  # Z
])
FIGURES -= np.array([2, 2])  # move figure anchor into center of figure (simplifies rotations)

ROTATION_90 = get_2d_rotation_matrix(-90).astype("int")
ROTATION_180 = get_2d_rotation_matrix(-180).astype("int")
ROTATION_270 = get_2d_rotation_matrix(-270).astype("int")

NINTENDO_SCORES = [0, 40, 100, 300, 1200]
SCORE_NORMALIZATION = 120

FIELD_SZ = 30

# Grid indices
STRUCTURE_IDX = 0
FIGURE_IDX = 1

MAX_TRAJECTORY_LENGTH = 100000  # Required because only completed episodes are saved in statistics

STYLE = {"bg_color": (50, 50, 50),
         "field_color": (70, 70, 70),
         "border_color": (130, 130, 130),
         "falling_color": [255, 255, 0],
         "grounded_color": [50, 200, 255],
         "font_color": (200, 200, 200),
         "font": "freesansbold.ttf"}


class Tetris(ParallelEnvironment):
    """An efficient Tetris simulator for simultaneous Tetris simulation with optional GUI."""
    NAME = "tetris"
    LEVELS = False
    TIME_RELEVANT = True
    WINS_RELEVANT = False

    FALLING_BLOCK_PADDING = 3  # prevents index out of bounds exceptions

    def __init__(self, num_par_inst=256, height=20, width=10):
        actions = ['IDLE', 'MOVE_LEFT', 'MOVE_RIGHT', 'ROT_RIGHT']
        super().__init__(num_par_inst, actions)

        self.height = height
        self.width = width

        self.grid = np.zeros(shape=(self.num_par_inst, self.height, self.width, 2), dtype='bool')
        # self.grid_structure = np.zeros(shape=(self.num_par_inst, self.height, self.width), dtype='bool')
        # self.grid_figure = np.zeros(shape=(self.num_par_inst, self.height, self.width), dtype='bool')

        # self.figure_rasterized = np.zeros(shape=(self.num_par_inst, 4, 4), dtype='bool')
        # self.figure_type = np.zeros(shape=self.num_par_inst, dtype='int8')
        self.figure = np.zeros(shape=(self.num_par_inst, 4, 2), dtype='int8')
        self.figure_position = np.zeros(shape=(self.num_par_inst, 2), dtype='int8')  # anchor is top left
        # self.figure_orientation = np.zeros(shape=self.num_par_inst, dtype='int8')  # 0: 0°, 1: 90°, 2: 180°, 3: 270°

        self.SPAWN_COORDS = [0, self.width // 2]

        self.reset()

        # For rendering purposes
        self.screens = None
        if self.num_par_inst < 8:
            self.init_gui()

    def reset(self, ids=slice(None), **kwargs):
        super(Tetris, self).reset(ids)

        # self.grid_structure[ids] = 0
        # self.grid_figure[ids] = 0
        self.grid[ids] = 0
        self.figure[ids] = 0
        self.figure_position[ids] = 0
        # self.figure_orientation[ids] = 0

        self.spawn_figure(ids)

    def get_structure_grid(self):
        return self.grid[..., STRUCTURE_IDX]

    def get_figure_grid(self):
        return self.grid[..., FIGURE_IDX]

    def spawn_figure(self, ids=slice(None)):
        """Replaces previous figure by a new random figure, spawned at the top of the grid.
        Also updates the corresponding figure grids and identifies situations where the new
        spawned figure spawned inside the already existing structure, leading to a game over."""
        if isinstance(ids, slice):
            num_new_figures = self.num_par_inst
        else:
            assert isinstance(ids, (np.ndarray, Sequence))
            num_new_figures = len(ids)

        new_figures_types = np.random.choice(len(FIGURES), size=num_new_figures, replace=True)
        new_figures = FIGURES[new_figures_types]

        # Register new figures
        self.figure[ids] = new_figures
        self.figure_position[ids] = self.SPAWN_COORDS
        # self.figure_orientation[ids] = 0

        self.update_figure_grid()

        # Identify game overs
        spawned_inside_block = np.any(self.get_structure_grid()[ids] &
                                      self.get_figure_grid()[ids], axis=(1, 2))
        if np.any(spawned_inside_block):
            if isinstance(ids, slice):
                ids = np.arange(self.num_par_inst)
            spawned_inside_block_ids = ids[spawned_inside_block]
            self.game_overs[spawned_inside_block_ids] = True

    def step(self, actions):
        old_scores = self.scores.copy()

        self.perform_actions(actions)  # takes 0.9 s (6 %) of computation time

        touchdown_ids = self.gravity()  # takes 9.0 s (58 %) of computation time

        structure_height_diff, hole_count_diff = self.handle_touchdowns(touchdown_ids)

        rewards = (self.scores - old_scores) / SCORE_NORMALIZATION

        # REWARD SHAPING
        # Penalize any increase of structure height (measured in number of occupied rows)
        rewards[touchdown_ids] -= np.maximum(structure_height_diff, 0) * 40 / SCORE_NORMALIZATION
        # Penalize roofed/covered empty fields and reward uncovered empty fields
        rewards[touchdown_ids] -= hole_count_diff * 10 / SCORE_NORMALIZATION
        # Penalize death
        rewards[self.game_overs] = -200 / SCORE_NORMALIZATION

        self.game_overs |= self.times >= MAX_TRAJECTORY_LENGTH
        self.times[~ self.game_overs] += 1

        # Reward mapping to [-1, +1]
        rewards = sigmoid(rewards)

        return rewards, self.scores, self.game_overs, self.times, self.wins, self.game_overs

    def perform_actions(self, actions):
        """Applies the given actions but does NOT update the figure grid."""
        move_left_ids = np.where(actions == 1)[0]
        move_right_ids = np.where(actions == 2)[0]
        rotate_right_ids = np.where(actions == 3)[0]

        if len(move_left_ids):
            self.move_figure_sidewise(move_left_ids, -1)
        if len(move_right_ids):
            self.move_figure_sidewise(move_right_ids, 1)
        if len(rotate_right_ids):
            self.rotate_figure_clockwise(rotate_right_ids)

    def gravity(self):
        """Performs a gravity step, i.e., lets fall the falling block by one step if possible.
        Returns a list of all instance IDs where the falling block touched the ground.
        Also updates the figure grids.

        :return The IDs of those environments where the figure touched the ground."""
        figure_field_coordinates = self.get_figure_field_coordinates()

        # Hypothetically, move all falling blocks down by one step
        figure_field_coordinates[:, :, 0] += 1

        # Identify touchdowns
        collides_floor = self.collides_floor(figure_field_coordinates)
        touchdown = collides_floor
        if np.any(~collides_floor):
            collides_structure = self.collides_structure(figure_field_coordinates[~collides_floor],
                                                         ids=np.where(~collides_floor)[0])
            touchdown[~collides_floor] = collides_structure

        # Identify touchdowns that lead to game over
        still_entering = self.entering(figure_field_coordinates)

        self.game_overs |= touchdown & still_entering

        # Apply actual gravity step only to still falling figures
        self.figure_position[~touchdown, 0] += 1

        self.update_figure_grid()

        touchdown_ids = np.where(touchdown)[0] if np.any(touchdown) else []
        return touchdown_ids

    def handle_touchdowns(self, ids):
        structure_height = self.get_structure_heights(ids)
        roofed_fields_count = self.count_holes(ids)

        self.place_falling_block(ids)
        self.rows_to_points(ids)
        self.spawn_figure(ids)

        structure_height_diff = self.get_structure_heights(ids) - structure_height
        hole_count_diff = self.count_holes(ids) - roofed_fields_count

        return structure_height_diff, hole_count_diff

    def rows_to_points(self, ids):
        """Identifies all full rows, deletes them, moves down the upper rows and turns
        the completed rows into points added to the score."""

        if len(ids) == 0:
            return

        # Find full rows and count them
        full_rows = np.all(self.get_structure_grid()[ids], axis=2)
        full_rows_counts = full_rows.sum(axis=1)
        any_full_row = np.where(np.any(full_rows, axis=1))[0]

        # Let rows above cleared rows fall down
        for i, idx in zip(any_full_row, ids[any_full_row]):
            for full_row_id in np.where(full_rows[i])[0]:
                self.get_structure_grid()[idx, 1:full_row_id + 1] = self.get_structure_grid()[idx, 0:full_row_id]
                self.get_structure_grid()[idx, 0] = False

        # Compute gained points from completed rows and add them to the scores
        points = np.take(NINTENDO_SCORES, indices=full_rows_counts)
        self.scores[ids] += points

    def place_falling_block(self, ids):
        """Places the falling block into the Tetris field."""
        # Place the falling block in all given envs
        self.get_structure_grid()[ids] = self.get_structure_grid()[ids] | self.get_figure_grid()[ids]

    def update_figure_grid(self, ids=slice(None)):
        # Reset grid
        self.get_figure_grid()[ids] = 0

        # Fetch current figure field coordinates
        figure_field_coordinates = self.get_figure_field_coordinates(ids=ids)

        # Identify those figure fields that are inside the grid
        inside = ~self.field_outside_grid(figure_field_coordinates)

        # Convert slice to proper ids range
        ids = np.arange(self.num_par_inst) if isinstance(ids, slice) else ids
        ids = np.expand_dims(ids, axis=1)

        # Apply to grid
        self.get_figure_grid()[
            ids,
            figure_field_coordinates[:, :, 0],
            figure_field_coordinates[:, :, 1]
        ] = inside

    def get_structure_heights(self, ids):
        occupied_row_indicators = np.any(self.get_structure_grid()[ids], axis=2)
        num_occupied_rows = np.sum(occupied_row_indicators, axis=1)
        return num_occupied_rows

    def count_holes(self, ids):
        """Returns the number of fields that are topped by a field."""
        roofed_indicators = np.zeros((len(ids), self.width), dtype='bool')
        roofed_field_counts_per_column = np.zeros((len(ids), self.width), dtype='int')
        for h in range(self.height):
            row = self.get_structure_grid()[ids, h]
            roofed_indicators |= row
            roofed_field_counts_per_column[roofed_indicators] += ~row[roofed_indicators]
        return np.sum(roofed_field_counts_per_column, axis=1)

    def get_states(self):
        """Returns a "k x 2 x w x h" matrix representing the Tetris grid environment.
        Empty fields are represented by False, all others by True."""

        # states = np.stack((self.get_structure_grid(), self.get_figure_grid()), axis=3)  # TODO: Combine both arrays into one
        return [self.grid.copy()]

    def get_state_shapes(self):
        image_state_shape = (self.height, self.width, 2)
        return [image_state_shape]

    def get_figure_field_coordinates(self, ids=slice(None)):
        figure = self.figure[ids]

        # Apply positions
        return figure + np.expand_dims(self.figure_position[ids], axis=1)

    def move_figure_sidewise(self, ids, delta: int):
        """Moves the figure to the right if delta > 0, otherwise to the left."""
        figure_field_coordinates = self.get_figure_field_coordinates(ids=ids)

        # Do hypothetical move
        figure_field_coordinates[:, :, 1] += delta

        # Identify illegal moves
        collides_with_wall = self.collides_left_wall(figure_field_coordinates) \
                             | self.collides_right_wall(figure_field_coordinates)
        is_illegal = collides_with_wall
        if np.any(~collides_with_wall):
            collides_with_structure = self.collides_structure(figure_field_coordinates[~collides_with_wall],
                                                              ids=ids[~collides_with_wall])
            is_illegal[~collides_with_wall] = collides_with_structure

        # Filter out illegal moves
        legal_ids = ids[~is_illegal]

        # Apply only legal moves
        self.figure_position[legal_ids, 1] += delta

    def rotate_figure_clockwise(self, ids):
        figure_original = self.figure[ids].copy()

        # Apply rotation
        self.figure[ids] = np.matmul(self.figure[ids], ROTATION_90)

        # Get resulting figure field coordinates
        figure_field_coordinates = self.get_figure_field_coordinates(ids)

        # Check for any invalid positions
        collides_left_wall = self.collides_left_wall(figure_field_coordinates)
        collides_right_wall = self.collides_right_wall(figure_field_coordinates)
        collides_floor = self.collides_floor(figure_field_coordinates)
        collides_wall = collides_left_wall | collides_right_wall | collides_floor
        collides = collides_wall
        if np.any(~collides_wall):
            collides_structure = self.collides_structure(figure_field_coordinates[~collides_wall],
                                                         ids=ids[~collides_wall])
            collides[~collides_wall] |= collides_structure
        collision_ids = ids[collides]

        # Undo rotation for collision cases
        self.figure[collision_ids] = figure_original[collides]

    def collides(self, figure_field_coordinates, ids=slice(None)):
        """Takes (hypothetical) field coordinates, the corresponding environment IDs and checks if
        the corresponding fields collide with a wall or the structure. If ids is not provided, then
        assumes to check all environments for collisions."""
        collides_left_wall = self.collides_left_wall(figure_field_coordinates)
        collides_right_wall = self.collides_right_wall(figure_field_coordinates)
        collides_structure = self.collides_structure(figure_field_coordinates, ids=ids)
        return collides_left_wall | collides_right_wall | collides_structure

    def collides_left_wall(self, figure_field_coordinates):
        return np.min(figure_field_coordinates[:, :, 1], axis=1) < 0

    def collides_right_wall(self, figure_field_coordinates):
        return np.max(figure_field_coordinates[:, :, 1], axis=1) >= self.width

    def collides_floor(self, figure_field_coordinates):
        return np.max(figure_field_coordinates[:, :, 0], axis=1) >= self.height

    def entering(self, figure_field_coordinates):
        return np.min(figure_field_coordinates[:, :, 0], axis=1) < 0

    def collides_structure(self, figure_field_coordinates, ids=slice(None)):
        ids = np.arange(self.num_par_inst) if isinstance(ids, slice) else ids
        ids = np.expand_dims(ids, axis=1)
        # Identify figure fields that stick out of the grid
        outside = self.field_outside_grid(figure_field_coordinates)
        return np.any(self.get_structure_grid()[
                          ids,
                          figure_field_coordinates[:, :, 0],
                          figure_field_coordinates[:, :, 1]
                      ],
                      axis=1,
                      where=~outside)

    def field_outside_grid(self, field_coordinates):
        return (field_coordinates[:, :, 1] < 0) | \
            (field_coordinates[:, :, 1] >= self.width) | \
            (field_coordinates[:, :, 0] < 0) | \
            (field_coordinates[:, :, 0] >= self.height)

    def init_gui(self):
        pygame.init()
        pygame.display.set_caption("Tetris")
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.num_par_inst * (50 + self.width * FIELD_SZ),
                                               240 + self.height * FIELD_SZ))
        # self.clock = pygame.time.Clock()

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                pass

        self.screen.fill(STYLE['bg_color'])

        scr_marg_left = 40
        scr_marg_top = 120

        # Title text
        font = pygame.font.SysFont('Consolas', 50)
        text = font.render("Tetris", True, STYLE["font_color"], None)
        rect = text.get_rect()
        rect.bottomleft = (scr_marg_left, scr_marg_top - 30)
        self.screen.blit(text, rect)

        # render environment instances
        for instance_id in range(self.num_par_inst):
            env_marg_left = 40 + instance_id * (FIELD_SZ * self.width + 2 + 40)

            env_marg_top = scr_marg_top

            area_width = FIELD_SZ * self.width + 2
            area_height = FIELD_SZ * self.height + 2

            grid_marg_left = env_marg_left
            grid_marg_top = env_marg_top + 80

            # Grid area
            pygame.draw.rect(self.screen, STYLE['border_color'],
                             [grid_marg_left - 2, grid_marg_top - 2, area_width, area_height])

            # Empty fields
            for y in range(self.height):
                for x in range(self.width):
                    pygame.draw.rect(self.screen, STYLE['field_color'],
                                     [grid_marg_left + x * FIELD_SZ, grid_marg_top + y * FIELD_SZ,
                                      FIELD_SZ - 2, FIELD_SZ - 2])

            # Falling figure
            rasterized_figure = np.where(self.get_figure_grid()[instance_id])
            for y, x in zip(rasterized_figure[0], rasterized_figure[1]):
                pygame.draw.rect(self.screen, STYLE['falling_color'],
                                 [grid_marg_left + x * FIELD_SZ, grid_marg_top + y * FIELD_SZ,
                                  FIELD_SZ - 2, FIELD_SZ - 2])

            # Lying blocks
            field_coords = np.where(self.get_structure_grid()[instance_id])
            for y, x in zip(field_coords[0], field_coords[1]):
                pygame.draw.rect(self.screen, STYLE['grounded_color'],
                                 [grid_marg_left + x * FIELD_SZ, grid_marg_top + y * FIELD_SZ,
                                  FIELD_SZ - 2, FIELD_SZ - 2])

            # Score text
            text, rect = self._place_text("Score")
            rect.topright = (env_marg_left + area_width, env_marg_top)
            self.screen.blit(text, rect)

            text, rect = self._place_text(str(self.scores[instance_id]))
            rect.topright = (env_marg_left + area_width, env_marg_top + 40)
            self.screen.blit(text, rect)

            # Time text
            text, rect = self._place_text("Time")
            rect.topleft = (env_marg_left, env_marg_top)
            self.screen.blit(text, rect)

            text, rect = self._place_text(str(self.times[instance_id]))
            rect.topleft = (env_marg_left, env_marg_top + 40)
            self.screen.blit(text, rect)

        pygame.display.flip()
        # self.clock.tick(self.speed)

    def _place_text(self, string: str, font_size:int=30):
        font = pygame.font.SysFont('Consolas', font_size)
        text = font.render(string, True, STYLE["font_color"], None)
        return text, text.get_rect()

    def state2text(self, state):
        field = state[0]
        return self.state_2d_to_text(field)

    def state_2d_to_text(self, state):
        lying_block_grid = state[:, :, 0]
        falling_block_grid = state[:, :, 1]
        grid_width = state.shape[1]
        grid_height = state.shape[0]

        text = ""

        text += "--" * (grid_width + 2) + "-\n"
        for row in range(grid_height):
            text += " |"
            for col in range(grid_width):
                if lying_block_grid[row, col]:
                    text += " #"
                elif falling_block_grid[row, col]:
                    text += " %"
                else:
                    text += "  "
            text += " |\n"
        text += "--" * (grid_width + 2) + "-"

        return text

    def get_config(self):
        config = {"height": self.height,
                  "width": self.width}
        config.update(super(Tetris, self).get_config())
        return config
