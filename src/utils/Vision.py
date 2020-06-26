from src.computer_vision.GroundTruthReader import GroundTruthReader
import json
import numpy as np
import logging


class Vision:
    """Wrapper around GroundTruthReader to provide some simple functions like
    extracting the slingshot reference point. Mainly copied from naive agent
    and TrajectoryPlanner."""

    def __init__(self):
        # for calculating reference point
        self.X_OFFSET = 0.45
        self.Y_OFFSET = 0.35
        self.scale_factor = 2.7

        # initialise colormap for the ground truth reader
        f = open('src/computer_vision/ColorMap.json', 'r')
        result = json.load(f)

        self.look_up_matrix = np.zeros((len(result), 256))
        self.look_up_obj_type = np.zeros(len(result)).astype(str)

        obj_number = 0
        for d in result:

            if 'effects_21' in d['type']:
                obj_name = 'Platform'

            elif 'effects_34' in d['type']:
                obj_name = 'TNT'

            elif 'ice' in d['type']:
                obj_name = 'Ice'

            elif 'wood' in d['type']:
                obj_name = 'Wood'

            elif 'stone' in d['type']:
                obj_name = 'Stone'

            else:
                obj_name = d['type'][:-2]

            obj_color_map = d['colormap']

            self.look_up_obj_type[obj_number] = obj_name
            for pair in obj_color_map:
                self.look_up_matrix[obj_number][int(pair['x'])] = pair['y']

            obj_number += 1

        # normalise the look_up_matrix
        self.look_up_matrix = self.look_up_matrix / np.sqrt((self.look_up_matrix ** 2).sum(1)).reshape(-1, 1)
        self._logger = logging.getLogger("Vision")

        self.image = None
        self.ground_truth = None

        self.gtr = None  # the GroundTruthReader

    def update(self, image, ground_truth):
        self.image = image
        self.ground_truth = ground_truth

    def get_sling_reference(self):
        """Returns the sling reference point."""

        self.gtr = GroundTruthReader(self.ground_truth, self.look_up_matrix, self.look_up_obj_type)
        self.gtr.set_screenshot(self.image)

        # Obtain slingshot
        sling = self.gtr.find_slingshot_mbr()[0]

        # Hotfix for a bug from original code
        sling.width, sling.height = sling.height, sling.width

        if sling is None:
            print("No sling found! Using default sling reference point.")
            x = 191
            y = 344

        else:
            x = int(sling.X + self.X_OFFSET * sling.width)
            y = int(sling.Y + self.Y_OFFSET * sling.width)

        return x, y
