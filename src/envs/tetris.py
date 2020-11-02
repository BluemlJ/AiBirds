import pygame
import numpy as np
import json

from src.envs.env import ParallelEnvironment

BLOCK_TYPES = {
    'I': np.array([[2, 0], [2, 1], [2, 2], [2, 3]]),
    'J': np.array([[0, 2], [1, 2], [2, 2], [2, 1]]),
    'L': np.array([[0, 1], [1, 1], [2, 1], [2, 2]]),
    'O': np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
    'S': np.array([[1, 2], [2, 2], [2, 1], [3, 1]]),
    'T': np.array([[2, 1], [2, 2], [2, 3], [3, 2]]),
    'Z': np.array([[2, 1], [2, 2], [1, 2], [1, 3]]),
}

BLOCK_TYPES_TENSOR = np.zeros(shape=(len(BLOCK_TYPES), 4, 4), dtype='bool')

for i, block_type in enumerate(BLOCK_TYPES):
    for y, x in BLOCK_TYPES[block_type]:
        BLOCK_TYPES_TENSOR[i, y, x] = True

NINTENDO_SCORES = [0, 40, 100, 300, 1200]
SCORE_NORMALIZATION = 40


class Tetris(ParallelEnvironment):
    """An efficient Tetris simulator for simultaneous Tetris simulation with optional GUI."""

    def __init__(self, num_par_envs=256, height=20, width=10, style_name="minimal"):
        actions = ['IDLE', 'MOVE_LEFT', 'MOVE_RIGHT', 'ROT_RIGHT']
        super().__init__("tetris", num_par_envs, actions)

        self.height = height
        self.width = width

        self.SPAWN_COORDS = [-2, self.width // 2 - 2]

        self.zoom = 20
        self.speed = 500

        self.game_overs = None
        self.fields = None
        self.times = None
        self.scores = None

        # Falling block properties
        self.fb_fields = None
        self.fb_shapes = None
        self.fb_anchors = None
        self.FB_PADDING = 3  # prevents index out of bounds exceptions

        self.reset()

        # For rendering purposes
        self.screen = None

        if self.num_par_envs < 8:
            self.init_gui()

        self.style_name = style_name
        with open('src/envs/style.json', 'r') as f:
            self.style = json.load(f)[self.style_name]

    def reset(self):
        self.game_overs = np.asarray(self.num_par_envs * [False])
        self.fields = np.zeros(shape=(self.num_par_envs, self.height, self.width), dtype='bool')
        self.times = np.zeros(shape=self.num_par_envs)
        self.scores = np.zeros(shape=self.num_par_envs)

        # Falling block properties
        self.fb_fields = np.zeros(shape=(self.num_par_envs, self.height, self.width), dtype='bool')
        self.fb_shapes = np.zeros(shape=(self.num_par_envs, 4, 4), dtype='bool')
        self.fb_anchors = np.zeros(shape=(self.num_par_envs, 2), dtype='int8')

        self.spawn_falling_blocks()

    def reset_for(self, ids):
        self.game_overs[ids] = False
        self.fields[ids] = 0
        self.times[ids] = 0
        self.scores[ids] = 0

        self.fb_fields[ids] = 0
        self.fb_shapes[ids] = 0
        self.fb_anchors[ids] = 0

        self.spawn_falling_blocks(ids)

    def spawn_falling_blocks(self, ids=None):
        if ids is None:
            ids = np.arange(self.num_par_envs)

        self.fb_shapes[ids] = BLOCK_TYPES_TENSOR[np.random.choice(len(BLOCK_TYPES_TENSOR), size=len(ids), replace=True)]
        self.fb_anchors[ids] = self.SPAWN_COORDS

        self.compute_fb_fields(ids)

        spawned_inside_block = np.any(self.fields[ids] & self.fb_fields[ids], axis=(1, 2))
        spawned_inside_block_ids = ids[spawned_inside_block]
        self.game_overs[spawned_inside_block_ids] = True

    def step(self, actions):
        old_scores = self.scores.copy()  # 1 ms

        self.perform_actions(actions)

        grounded_ids = self.gravity()

        self.compute_fb_fields(range(self.num_par_envs))

        if len(grounded_ids):
            self.place_falling_block(grounded_ids)  # 1 ms
            self.rows_to_points(grounded_ids)  # 3 ms
            self.spawn_falling_blocks(grounded_ids)  # 6 ms

        self.times[~ self.game_overs] += 1

        states = self.get_states()

        rewards = (self.scores - old_scores) / SCORE_NORMALIZATION
        # rewards[self.game_overs] = -40 / SCORE_NORMALIZATION

        return states, rewards, self.scores, self.game_overs, self.times

    def perform_actions(self, actions):
        move_left_ids = np.where(actions == 1)[0]
        move_right_ids = np.where(actions == 2)[0]
        rotate_right_ids = np.where(actions == 3)[0]

        if len(move_left_ids):
            self.move_figure(move_left_ids, -1)
        if len(move_right_ids):
            self.move_figure(move_right_ids, 1)
        if len(rotate_right_ids):
            self.rotate_figure(rotate_right_ids, 1)

    def rows_to_points(self, ids):
        """Identifies all full rows, deletes them, moves down the upper rows and turns
        the completed rows into points added to the score."""

        # Find full rows and count them
        full_rows = np.all(self.fields[ids], axis=2)
        full_rows_counts = full_rows.sum(axis=1)
        any_full_row = np.where(np.any(full_rows, axis=1))[0]

        # Let rows above cleared rows fall down
        for i, idx in zip(any_full_row, ids[any_full_row]):
            for full_row_id in np.where(full_rows[i])[0]:
                self.fields[idx, 1:full_row_id + 1] = self.fields[idx, 0:full_row_id]
                self.fields[idx, 0] = False

        # Compute gained points from completed rows and add them to the scores
        points = np.take(NINTENDO_SCORES, indices=full_rows_counts)
        self.scores[ids] += points

    def gravity(self):
        """Performs a gravity step, i.e., lets fall the falling block by one step if possible. Returns a list of
        all instance IDs where the falling block grounded."""

        # Move all falling blocks down by one step
        self.fb_anchors[:, 0] += 1

        # Check if any falling block grounded, i.e., touched a lying block or touched the ground
        _, grounded, still_entering = self.check_for_collisions(range(self.num_par_envs))  # 104 ms

        self.game_overs = grounded & still_entering | self.game_overs

        # Undo down move for grounded falling blocks
        grounded_ids = np.where(grounded)[0]
        self.fb_anchors[grounded_ids, 0] -= 1

        return grounded_ids

    def place_falling_block(self, ids):
        """Places the falling block into the Tetris field."""
        # Place the falling block in all given envs
        self.fields[ids] = self.fields[ids] | self.fb_fields[ids]

    def compute_fb_fields(self, ids):
        # get padded field
        padded_fb_fields = self.get_padded_fb_fields(ids)

        # unpad (and get rid of invalid positions)
        self.fb_fields[ids] = padded_fb_fields[:, self.FB_PADDING:-self.FB_PADDING, self.FB_PADDING:-self.FB_PADDING]

    def get_padded_fb_fields(self, ids):
        num_ids = len(ids)

        # pad for convenience
        padded_fb_fields = np.zeros(shape=(num_ids, self.height + 2*self.FB_PADDING, self.width + 2*self.FB_PADDING),
                                    dtype='bool')

        anchor_ys = self.fb_anchors[ids, 0] + self.FB_PADDING
        anchor_xs = self.fb_anchors[ids, 1] + self.FB_PADDING

        range_mat = np.array([np.arange(4)] * num_ids)

        z = np.expand_dims(np.arange(num_ids), axis=(1, 2))
        y = np.expand_dims(anchor_ys[:, None] + range_mat, axis=2)
        x = np.expand_dims(anchor_xs[:, None] + range_mat, axis=1)

        padded_fb_fields[tuple((z, y, x))] = self.fb_shapes[ids]

        return padded_fb_fields

    def get_states(self):
        """Returns a "k x 2 x w x h" matrix representing the Tetris grid environment.
        Empty fields are represented by False, all others by True."""

        states = np.stack((self.fields, self.fb_fields), axis=3)
        return states, []

    def get_state_shape(self):
        image_state_shape = [self.height, self.width, 2]
        numerical_state_shape = 0
        return image_state_shape, numerical_state_shape

    def rotate_figure(self, ids, direction):
        shapes_old = self.fb_shapes[ids].copy()

        # Rotate the shapes
        self.fb_shapes[ids] = np.rot90(self.fb_shapes[ids], axes=(1,2), k=direction)

        collisions, _, _ = self.check_for_collisions(ids)
        collisions_ids = ids[np.where(collisions)[0]]

        # Undo rotation for collision cases
        self.fb_shapes[collisions_ids] = shapes_old[np.where(collisions)[0]]

    def move_figure(self, ids, delta):
        self.fb_anchors[ids, 1] += delta

        collisions, _, _ = self.check_for_collisions(ids)
        collisions_ids = ids[np.where(collisions)[0]]

        # Undo move for collision cases
        self.fb_anchors[collisions_ids, 1] -= delta

    def check_for_collisions(self, ids):
        """Returns two boolean matrices. The first matrix tells for each parallel instance of
        instance_ids if a collision happened. The second matrix tells for each parallel instance
        if the falling block is still entering the field."""
        # TODO: efficiency, takes 104 ms for 4000 instances

        padded_fb_fields = self.get_padded_fb_fields(ids)  # 4 ms

        # Rest 3 ms
        still_entering = np.any(padded_fb_fields[:, :self.FB_PADDING + 1, :], axis=(1, 2))

        collides_other_block = np.any(
            padded_fb_fields[:, self.FB_PADDING:-self.FB_PADDING, self.FB_PADDING:-self.FB_PADDING] & self.fields[ids],
            axis=(1, 2))

        collides_left_wall = np.any(padded_fb_fields[:, :, self.FB_PADDING - 1], axis=1)
        collides_right_wall = np.any(padded_fb_fields[:, :, -(self.FB_PADDING)], axis=1)

        collides_ground = np.any(padded_fb_fields[:, -self.FB_PADDING, :], axis=1)

        collides = collides_other_block | collides_ground | collides_left_wall | collides_right_wall
        grounded = collides_other_block | collides_ground

        return collides, grounded, still_entering

    def init_gui(self):
        pygame.init()
        pygame.display.set_caption("Tetris2D")
        self.screen = pygame.display.set_mode((100 + self.width * self.zoom, 150 + self.height * self.zoom))
        self.clock = pygame.time.Clock()

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                pass

        self.screen.fill(self.style['bg_color'])

        # render environment instances
        for instance_id in range(self.num_par_envs):

            x_margin = (self.screen.get_width() - self.num_par_envs * (self.zoom * self.width) - (
                    self.num_par_envs - 1) * self.zoom) // 2 + instance_id * (
                                   self.zoom * self.width + self.zoom)
            y_margin = self.zoom

            # render figure
            fb_coords = np.where(self.fb_fields[instance_id])
            for y, x in zip(fb_coords[0], fb_coords[1]):
                pygame.draw.rect(self.screen, self.style['falling_color'],
                                 [x_margin + x * self.zoom, y_margin + y * self.zoom, self.zoom, self.zoom]),

            # render field blocks
            field_coords = np.where(self.fields[instance_id])
            for y, x in zip(field_coords[0], field_coords[1]):
                pygame.draw.rect(self.screen, self.style['grounded_color'],
                                 [x_margin + x * self.zoom, y_margin + y * self.zoom, self.zoom, self.zoom]),

            # render grid lines
            for y in range(self.height):
                for x in range(self.width):
                    pygame.draw.rect(self.screen, self.style['line_color'],
                                     [x_margin + x * self.zoom, y_margin + y * self.zoom, self.zoom, self.zoom], 1)

            # write score to screen
            font = pygame.font.Font(self.style['font'], self.zoom)
            text = font.render(f'score: {self.scores[instance_id]}', True, (0, 0, 0), (255, 255, 255))
            rect = text.get_rect()
            rect.center = (x_margin + self.width // 2 * self.zoom, y_margin - self.zoom // 2 - 1)

            self.screen.blit(text, rect)

        pygame.display.flip()
        self.clock.tick(self.speed)

    def image_state_to_text(self, state):
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