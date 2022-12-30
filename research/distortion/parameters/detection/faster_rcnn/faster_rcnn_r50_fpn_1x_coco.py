
class Params:
    def __init__(self):
        self.BLOCK_NAMES = \
            [
                'stem',
                'layer1_0',
                'layer1_1',
                'layer1_2',
                'layer2_0',
                'layer2_1',
                'layer2_2',
                'layer2_3',
                'layer3_0',
                'layer3_1',
                'layer3_2',
                'layer3_3',
                'layer3_4',
                'layer3_5',
                'layer4_0',
                'layer4_1',
                'layer4_2',
                None
            ]

        self.LAYER_NAME_TO_BLOCK_NAME = \
            {
                "stem": "stem",
                "layer1_0_1": "layer1_0",
                "layer1_0_2": "layer1_0",
                "layer1_0_3": "layer1_0",
                "layer1_1_1": "layer1_1",
                "layer1_1_2": "layer1_1",
                "layer1_1_3": "layer1_1",
                "layer1_2_1": "layer1_2",
                "layer1_2_2": "layer1_2",
                "layer1_2_3": "layer1_2",
                "layer2_0_1": "layer2_0",
                "layer2_0_2": "layer2_0",
                "layer2_0_3": "layer2_0",
                "layer2_1_1": "layer2_1",
                "layer2_1_2": "layer2_1",
                "layer2_1_3": "layer2_1",
                "layer2_2_1": "layer2_2",
                "layer2_2_2": "layer2_2",
                "layer2_2_3": "layer2_2",
                "layer2_3_1": "layer2_3",
                "layer2_3_2": "layer2_3",
                "layer2_3_3": "layer2_3",
                "layer3_0_1": "layer3_0",
                "layer3_0_2": "layer3_0",
                "layer3_0_3": "layer3_0",
                "layer3_1_1": "layer3_1",
                "layer3_1_2": "layer3_1",
                "layer3_1_3": "layer3_1",
                "layer3_2_1": "layer3_2",
                "layer3_2_2": "layer3_2",
                "layer3_2_3": "layer3_2",
                "layer3_3_1": "layer3_3",
                "layer3_3_2": "layer3_3",
                "layer3_3_3": "layer3_3",
                "layer3_4_1": "layer3_4",
                "layer3_4_2": "layer3_4",
                "layer3_4_3": "layer3_4",
                "layer3_5_1": "layer3_5",
                "layer3_5_2": "layer3_5",
                "layer3_5_3": "layer3_5",
                "layer4_0_1": "layer4_0",
                "layer4_0_2": "layer4_0",
                "layer4_0_3": "layer4_0",
                "layer4_1_1": "layer4_1",
                "layer4_1_2": "layer4_1",
                "layer4_1_3": "layer4_1",
                "layer4_2_1": "layer4_2",
                "layer4_2_2": "layer4_2",
                "layer4_2_3": "layer4_2",
            }

        self.LAYER_NAMES = \
            [
                "stem",
                "layer1_0_1",
                "layer1_0_2",
                "layer1_0_3",
                "layer1_1_1",
                "layer1_1_2",
                "layer1_1_3",
                "layer1_2_1",
                "layer1_2_2",
                "layer1_2_3",
                "layer2_0_1",
                "layer2_0_2",
                "layer2_0_3",
                "layer2_1_1",
                "layer2_1_2",
                "layer2_1_3",
                "layer2_2_1",
                "layer2_2_2",
                "layer2_2_3",
                "layer2_3_1",
                "layer2_3_2",
                "layer2_3_3",
                "layer3_0_1",
                "layer3_0_2",
                "layer3_0_3",
                "layer3_1_1",
                "layer3_1_2",
                "layer3_1_3",
                "layer3_2_1",
                "layer3_2_2",
                "layer3_2_3",
                "layer3_3_1",
                "layer3_3_2",
                "layer3_3_3",
                "layer3_4_1",
                "layer3_4_2",
                "layer3_4_3",
                "layer3_5_1",
                "layer3_5_2",
                "layer3_5_3",
                "layer4_0_1",
                "layer4_0_2",
                "layer4_0_3",
                "layer4_1_1",
                "layer4_1_2",
                "layer4_1_3",
                "layer4_2_1",
                "layer4_2_2",
                "layer4_2_3",
            ]

        self.BLOCK_INPUT_DICT = \
            {
                'stem': 'input_images',
                'layer1_0': 'stem',
                'layer1_1': 'layer1_0',
                'layer1_2': 'layer1_1',
                'layer2_0': 'layer1_2',
                'layer2_1': 'layer2_0',
                'layer2_2': 'layer2_1',
                'layer2_3': 'layer2_2',
                'layer3_0': 'layer2_3',
                'layer3_1': 'layer3_0',
                'layer3_2': 'layer3_1',
                'layer3_3': 'layer3_2',
                'layer3_4': 'layer3_3',
                'layer3_5': 'layer3_4',
                'layer4_0': 'layer3_5',
                'layer4_1': 'layer4_0',
                'layer4_2': 'layer4_1',
                None: 'layer4_2'
            }
        assert False
        self.LAYER_NAME_TO_DIMS = \
            {
                'stem': [64, 400, 672],
                'layer1_0_1': [64, 200, 336],
                'layer1_0_2': [64, 200, 336],
                'layer1_0_3': [256, 200, 336],
                'layer1_1_1': [64, 200, 336],
                'layer1_1_2': [64, 200, 336],
                'layer1_1_3': [256, 200, 336],
                'layer1_2_1': [64, 200, 336],
                'layer1_2_2': [64, 200, 336],
                'layer1_2_3': [256, 200, 336],
                'layer2_0_1': [128, 200, 336],
                'layer2_0_2': [128, 100, 168],
                'layer2_0_3': [512, 100, 168],
                'layer2_1_1': [128, 100, 168],
                'layer2_1_2': [128, 100, 168],
                'layer2_1_3': [512, 100, 168],
                'layer2_2_1': [128, 100, 168],
                'layer2_2_2': [128, 100, 168],
                'layer2_2_3': [512, 100, 168],
                'layer2_3_1': [128, 100, 168],
                'layer2_3_2': [128, 100, 168],
                'layer2_3_3': [512, 100, 168],
                'layer3_0_1': [256, 100, 168],
                'layer3_0_2': [256, 50, 84],
                'layer3_0_3': [1024, 50, 84],
                'layer3_1_1': [256, 50, 84],
                'layer3_1_2': [256, 50, 84],
                'layer3_1_3': [1024, 50, 84],
                'layer3_2_1': [256, 50, 84],
                'layer3_2_2': [256, 50, 84],
                'layer3_2_3': [1024, 50, 84],
                'layer3_3_1': [256, 50, 84],
                'layer3_3_2': [256, 50, 84],
                'layer3_3_3': [1024, 50, 84],
                'layer3_4_1': [256, 50, 84],
                'layer3_4_2': [256, 50, 84],
                'layer3_4_3': [1024, 50, 84],
                'layer3_5_1': [256, 50, 84],
                'layer3_5_2': [256, 50, 84],
                'layer3_5_3': [1024, 50, 84],
                'layer4_0_1': [512, 50, 84],
                'layer4_0_2': [512, 25, 42],
                'layer4_0_3': [2048, 25, 42],
                'layer4_1_1': [512, 25, 42],
                'layer4_1_2': [512, 25, 42],
                'layer4_1_3': [2048, 25, 42],
                'layer4_2_1': [512, 25, 42],
                'layer4_2_2': [512, 25, 42],
                'layer4_2_3': [2048, 25, 42]
            }