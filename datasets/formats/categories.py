
# Replica dataste
#####################################

# Things and stuff class names lists
stuff_class_names = [
            'no_class',
            'base-cabinet',
            'bathtub',
            'beam',
            'blinds',
            'cabinet',
            'curtain',
            'ceiling',
            'desk',
            'door',
            'faucet',
            'floor',
            'mat',
            'nightstand',
            'panel',
            'paper-towel',
            'pillar',
            'pipe',
            'plant-stand',
            'plate',
            'rack',
            'refrigerator',
            'shelf',
            'shower-stall',
            'sink',
            'stair',
            'table',
            'table-runner',
            'tv-stand',
            'utensil-holder',
            'vent',
            'wall',
            'wall-cabinet',
            'wall-plug',
            'wardrobe',
            'window',
            'rug',
            'logo',
            'set-of-clothing',
            'text',
            'anonymized-text',
            'plane',
            'non-plane',
            'lamp',
        ]

things_class_names =[
            'backpack',
            'basket',
            'beanbag',
            'bed',
            'bench',
            'bike',
            'bin',
            'blanket',
            'book',
            'bottle',
            'box',
            'bowl',
            'camera',
            'candle',
            'chair',
            'chopping-board',
            'clock',
            'cloth',
            'clothing',
            'coaster',
            'comforter',
            'computer-keyboard',
            'cup',
            'cushion',
            'cooktop',
            'countertop',
            'desk-organizer',
            'desktop-computer',
            'exercise-ball',
            'handbag',
            'hair-dryer',
            'handrail',
            'indoor-plant',
            'knife-block',
            'kitchen-utensil',
            'laptop',
            'major-appliance',
            'microwave',
            'monitor',
            'mouse',
            'pan',
            'phone',
            'picture',
            'pillow',
            'pot',
            'remote-control',
            'scarf',
            'sculpture',
            'shoe',
            'small-appliance',
            'sofa',
            'stool',
            'switch',
            'tablet',
            'tissue-paper',
            'toilet',
            'toothbrush',
            'towel',
            'tv-screen',
            'umbrella',
            'vase',
            'bag',
        ]

ADE20K_150_CATEGORIES = [
    {"color": [120, 120, 120], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [180, 120, 120], "id": 1, "isthing": 0, "name": "building"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "sky"},
    {"color": [80, 50, 50], "id": 3, "isthing": 0, "name": "floor"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "tree"},
    {"color": [120, 120, 80], "id": 5, "isthing": 0, "name": "ceiling"},
    {"color": [140, 140, 140], "id": 6, "isthing": 0, "name": "road, route"},
    {"color": [204, 5, 255], "id": 7, "isthing": 1, "name": "bed"},
    {"color": [230, 230, 230], "id": 8, "isthing": 1, "name": "window "},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "grass"},
    {"color": [224, 5, 255], "id": 10, "isthing": 1, "name": "cabinet"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "sidewalk, pavement"},
    {"color": [150, 5, 61], "id": 12, "isthing": 1, "name": "person"},
    {"color": [120, 120, 70], "id": 13, "isthing": 0, "name": "earth, ground"},
    {"color": [8, 255, 51], "id": 14, "isthing": 1, "name": "door"},
    {"color": [255, 6, 82], "id": 15, "isthing": 1, "name": "table"},
    {"color": [143, 255, 140], "id": 16, "isthing": 0, "name": "mountain, mount"},
    {"color": [204, 255, 4], "id": 17, "isthing": 0, "name": "plant"},
    {"color": [255, 51, 7], "id": 18, "isthing": 1, "name": "curtain"},
    {"color": [204, 70, 3], "id": 19, "isthing": 1, "name": "chair"},
    {"color": [0, 102, 200], "id": 20, "isthing": 1, "name": "car"},
    {"color": [61, 230, 250], "id": 21, "isthing": 0, "name": "water"},
    {"color": [255, 6, 51], "id": 22, "isthing": 1, "name": "painting, picture"},
    {"color": [11, 102, 255], "id": 23, "isthing": 1, "name": "sofa"},
    {"color": [255, 7, 71], "id": 24, "isthing": 1, "name": "shelf"},
    {"color": [255, 9, 224], "id": 25, "isthing": 0, "name": "house"},
    {"color": [9, 7, 230], "id": 26, "isthing": 0, "name": "sea"},
    {"color": [220, 220, 220], "id": 27, "isthing": 1, "name": "mirror"},
    {"color": [255, 9, 92], "id": 28, "isthing": 0, "name": "rug"},
    {"color": [112, 9, 255], "id": 29, "isthing": 0, "name": "field"},
    {"color": [8, 255, 214], "id": 30, "isthing": 1, "name": "armchair"},
    {"color": [7, 255, 224], "id": 31, "isthing": 1, "name": "seat"},
    {"color": [255, 184, 6], "id": 32, "isthing": 1, "name": "fence"},
    {"color": [10, 255, 71], "id": 33, "isthing": 1, "name": "desk"},
    {"color": [255, 41, 10], "id": 34, "isthing": 0, "name": "rock, stone"},
    {"color": [7, 255, 255], "id": 35, "isthing": 1, "name": "wardrobe, closet, press"},
    {"color": [224, 255, 8], "id": 36, "isthing": 1, "name": "lamp"},
    {"color": [102, 8, 255], "id": 37, "isthing": 1, "name": "tub"},
    {"color": [255, 61, 6], "id": 38, "isthing": 1, "name": "rail"},
    {"color": [255, 194, 7], "id": 39, "isthing": 1, "name": "cushion"},
    {"color": [255, 122, 8], "id": 40, "isthing": 0, "name": "base, pedestal, stand"},
    {"color": [0, 255, 20], "id": 41, "isthing": 1, "name": "box"},
    {"color": [255, 8, 41], "id": 42, "isthing": 1, "name": "column, pillar"},
    {"color": [255, 5, 153], "id": 43, "isthing": 1, "name": "signboard, sign"},
    {
        "color": [6, 51, 255],
        "id": 44,
        "isthing": 1,
        "name": "chest of drawers, chest, bureau, dresser",
    },
    {"color": [235, 12, 255], "id": 45, "isthing": 1, "name": "counter"},
    {"color": [160, 150, 20], "id": 46, "isthing": 0, "name": "sand"},
    {"color": [0, 163, 255], "id": 47, "isthing": 1, "name": "sink"},
    {"color": [140, 140, 140], "id": 48, "isthing": 0, "name": "skyscraper"},
    {"color": [250, 10, 15], "id": 49, "isthing": 1, "name": "fireplace"},
    {"color": [20, 255, 0], "id": 50, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [31, 255, 0], "id": 51, "isthing": 0, "name": "grandstand, covered stand"},
    {"color": [255, 31, 0], "id": 52, "isthing": 0, "name": "path"},
    {"color": [255, 224, 0], "id": 53, "isthing": 1, "name": "stairs"},
    {"color": [153, 255, 0], "id": 54, "isthing": 0, "name": "runway"},
    {"color": [0, 0, 255], "id": 55, "isthing": 1, "name": "case, display case, showcase, vitrine"},
    {
        "color": [255, 71, 0],
        "id": 56,
        "isthing": 1,
        "name": "pool table, billiard table, snooker table",
    },
    {"color": [0, 235, 255], "id": 57, "isthing": 1, "name": "pillow"},
    {"color": [0, 173, 255], "id": 58, "isthing": 1, "name": "screen door, screen"},
    {"color": [31, 0, 255], "id": 59, "isthing": 0, "name": "stairway, staircase"},
    {"color": [11, 200, 200], "id": 60, "isthing": 0, "name": "river"},
    {"color": [255, 82, 0], "id": 61, "isthing": 0, "name": "bridge, span"},
    {"color": [0, 255, 245], "id": 62, "isthing": 1, "name": "bookcase"},
    {"color": [0, 61, 255], "id": 63, "isthing": 0, "name": "blind, screen"},
    {"color": [0, 255, 112], "id": 64, "isthing": 1, "name": "coffee table"},
    {
        "color": [0, 255, 133],
        "id": 65,
        "isthing": 1,
        "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
    },
    {"color": [255, 0, 0], "id": 66, "isthing": 1, "name": "flower"},
    {"color": [255, 163, 0], "id": 67, "isthing": 1, "name": "book"},
    {"color": [255, 102, 0], "id": 68, "isthing": 0, "name": "hill"},
    {"color": [194, 255, 0], "id": 69, "isthing": 1, "name": "bench"},
    {"color": [0, 143, 255], "id": 70, "isthing": 1, "name": "countertop"},
    {"color": [51, 255, 0], "id": 71, "isthing": 1, "name": "stove"},
    {"color": [0, 82, 255], "id": 72, "isthing": 1, "name": "palm, palm tree"},
    {"color": [0, 255, 41], "id": 73, "isthing": 1, "name": "kitchen island"},
    {"color": [0, 255, 173], "id": 74, "isthing": 1, "name": "computer"},
    {"color": [10, 0, 255], "id": 75, "isthing": 1, "name": "swivel chair"},
    {"color": [173, 255, 0], "id": 76, "isthing": 1, "name": "boat"},
    {"color": [0, 255, 153], "id": 77, "isthing": 0, "name": "bar"},
    {"color": [255, 92, 0], "id": 78, "isthing": 1, "name": "arcade machine"},
    {"color": [255, 0, 255], "id": 79, "isthing": 0, "name": "hovel, hut, hutch, shack, shanty"},
    {"color": [255, 0, 245], "id": 80, "isthing": 1, "name": "bus"},
    {"color": [255, 0, 102], "id": 81, "isthing": 1, "name": "towel"},
    {"color": [255, 173, 0], "id": 82, "isthing": 1, "name": "light"},
    {"color": [255, 0, 20], "id": 83, "isthing": 1, "name": "truck"},
    {"color": [255, 184, 184], "id": 84, "isthing": 0, "name": "tower"},
    {"color": [0, 31, 255], "id": 85, "isthing": 1, "name": "chandelier"},
    {"color": [0, 255, 61], "id": 86, "isthing": 1, "name": "awning, sunshade, sunblind"},
    {"color": [0, 71, 255], "id": 87, "isthing": 1, "name": "street lamp"},
    {"color": [255, 0, 204], "id": 88, "isthing": 1, "name": "booth"},
    {"color": [0, 255, 194], "id": 89, "isthing": 1, "name": "tv"},
    {"color": [0, 255, 82], "id": 90, "isthing": 1, "name": "plane"},
    {"color": [0, 10, 255], "id": 91, "isthing": 0, "name": "dirt track"},
    {"color": [0, 112, 255], "id": 92, "isthing": 1, "name": "clothes"},
    {"color": [51, 0, 255], "id": 93, "isthing": 1, "name": "pole"},
    {"color": [0, 194, 255], "id": 94, "isthing": 0, "name": "land, ground, soil"},
    {
        "color": [0, 122, 255],
        "id": 95,
        "isthing": 1,
        "name": "bannister, banister, balustrade, balusters, handrail",
    },
    {
        "color": [0, 255, 163],
        "id": 96,
        "isthing": 0,
        "name": "escalator, moving staircase, moving stairway",
    },
    {
        "color": [255, 153, 0],
        "id": 97,
        "isthing": 1,
        "name": "ottoman, pouf, pouffe, puff, hassock",
    },
    {"color": [0, 255, 10], "id": 98, "isthing": 1, "name": "bottle"},
    {"color": [255, 112, 0], "id": 99, "isthing": 0, "name": "buffet, counter, sideboard"},
    {
        "color": [143, 255, 0],
        "id": 100,
        "isthing": 0,
        "name": "poster, posting, placard, notice, bill, card",
    },
    {"color": [82, 0, 255], "id": 101, "isthing": 0, "name": "stage"},
    {"color": [163, 255, 0], "id": 102, "isthing": 1, "name": "van"},
    {"color": [255, 235, 0], "id": 103, "isthing": 1, "name": "ship"},
    {"color": [8, 184, 170], "id": 104, "isthing": 1, "name": "fountain"},
    {
        "color": [133, 0, 255],
        "id": 105,
        "isthing": 0,
        "name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    },
    {"color": [0, 255, 92], "id": 106, "isthing": 0, "name": "canopy"},
    {
        "color": [184, 0, 255],
        "id": 107,
        "isthing": 1,
        "name": "washer, automatic washer, washing machine",
    },
    {"color": [255, 0, 31], "id": 108, "isthing": 1, "name": "plaything, toy"},
    {"color": [0, 184, 255], "id": 109, "isthing": 0, "name": "pool"},
    {"color": [0, 214, 255], "id": 110, "isthing": 1, "name": "stool"},
    {"color": [255, 0, 112], "id": 111, "isthing": 1, "name": "barrel, cask"},
    {"color": [92, 255, 0], "id": 112, "isthing": 1, "name": "basket, handbasket"},
    {"color": [0, 224, 255], "id": 113, "isthing": 0, "name": "falls"},
    {"color": [112, 224, 255], "id": 114, "isthing": 0, "name": "tent"},
    {"color": [70, 184, 160], "id": 115, "isthing": 1, "name": "bag"},
    {"color": [163, 0, 255], "id": 116, "isthing": 1, "name": "minibike, motorbike"},
    {"color": [153, 0, 255], "id": 117, "isthing": 0, "name": "cradle"},
    {"color": [71, 255, 0], "id": 118, "isthing": 1, "name": "oven"},
    {"color": [255, 0, 163], "id": 119, "isthing": 1, "name": "ball"},
    {"color": [255, 204, 0], "id": 120, "isthing": 1, "name": "food, solid food"},
    {"color": [255, 0, 143], "id": 121, "isthing": 1, "name": "step, stair"},
    {"color": [0, 255, 235], "id": 122, "isthing": 0, "name": "tank, storage tank"},
    {"color": [133, 255, 0], "id": 123, "isthing": 1, "name": "trade name"},
    {"color": [255, 0, 235], "id": 124, "isthing": 1, "name": "microwave"},
    {"color": [245, 0, 255], "id": 125, "isthing": 1, "name": "pot"},
    {"color": [255, 0, 122], "id": 126, "isthing": 1, "name": "animal"},
    {"color": [255, 245, 0], "id": 127, "isthing": 1, "name": "bicycle"},
    {"color": [10, 190, 212], "id": 128, "isthing": 0, "name": "lake"},
    {"color": [214, 255, 0], "id": 129, "isthing": 1, "name": "dishwasher"},
    {"color": [0, 204, 255], "id": 130, "isthing": 1, "name": "screen"},
    {"color": [20, 0, 255], "id": 131, "isthing": 0, "name": "blanket, cover"},
    {"color": [255, 255, 0], "id": 132, "isthing": 1, "name": "sculpture"},
    {"color": [0, 153, 255], "id": 133, "isthing": 1, "name": "hood, exhaust hood"},
    {"color": [0, 41, 255], "id": 134, "isthing": 1, "name": "sconce"},
    {"color": [0, 255, 204], "id": 135, "isthing": 1, "name": "vase"},
    {"color": [41, 0, 255], "id": 136, "isthing": 1, "name": "traffic light"},
    {"color": [41, 255, 0], "id": 137, "isthing": 1, "name": "tray"},
    {"color": [173, 0, 255], "id": 138, "isthing": 1, "name": "trash can"},
    {"color": [0, 245, 255], "id": 139, "isthing": 1, "name": "fan"},
    {"color": [71, 0, 255], "id": 140, "isthing": 0, "name": "pier"},
    {"color": [122, 0, 255], "id": 141, "isthing": 0, "name": "crt screen"},
    {"color": [0, 255, 184], "id": 142, "isthing": 1, "name": "plate"},
    {"color": [0, 92, 255], "id": 143, "isthing": 1, "name": "monitor"},
    {"color": [184, 255, 0], "id": 144, "isthing": 1, "name": "bulletin board"},
    {"color": [0, 133, 255], "id": 145, "isthing": 0, "name": "shower"},
    {"color": [255, 214, 0], "id": 146, "isthing": 1, "name": "radiator"},
    {"color": [25, 194, 194], "id": 147, "isthing": 1, "name": "glass, drinking glass"},
    {"color": [102, 255, 0], "id": 148, "isthing": 1, "name": "clock"},
    {"color": [92, 0, 255], "id": 149, "isthing": 1, "name": "flag"},
]

replica_ids_to_classes = {
  0: 'no_class',
  1: 'backpack',
  2: 'base-cabinet',
  3: 'basket',
  4: 'bathtub',
  5: 'beam',
  6: 'beanbag',
  7: 'bed',
  8: 'bench',
  9: 'bike',
  10: 'bin',
  11: 'blanket',
  12: 'blinds',
  13: 'book',
  14: 'bottle',
  15: 'box',
  16: 'bowl',
  17: 'camera',
  18: 'cabinet',
  19: 'candle',
  20: 'chair',
  21: 'chopping-board',
  22: 'clock',
  23: 'cloth',
  24: 'clothing',
  25: 'coaster',
  26: 'comforter',
  27: 'computer-keyboard',
  28: 'cup',
  29: 'cushion',
  30: 'curtain',
  31: 'ceiling',
  32: 'cooktop',
  33: 'countertop',
  34: 'desk',
  35: 'desk-organizer',
  36: 'desktop-computer',
  37: 'door',
  38: 'excercise-ball',
  39: 'faucet',
  40: 'floor',
  41: 'handbag',
  42: 'hair-dryer',
  43: 'handrail',
  44: 'indoor-plant',
  45: 'knife-block',
  46: 'kitchen-utensil',
  47: 'lamp',
  48: 'laptop',
  49: 'major-appliance',
  50: 'mat',
  51: 'microwave',
  52: 'monitor',
  53: 'mouse',
  54: 'nightstand',
  55: 'pan',
  56: 'panel',
  57: 'paper-towel',
  58: 'phone',
  59: 'picture',
  60: 'pillar',
  61: 'pillow',
  62: 'pipe',
  63: 'plant-stand',
  64: 'plate',
  65: 'pot',
  66: 'rack',
  67: 'refrigerator',
  68: 'remote-control',
  69: 'scarf',
  70: 'sculpture',
  71: 'shelf',
  72: 'shoe',
  73: 'shower-stall',
  74: 'sink',
  75: 'small-appliance',
  76: 'sofa',
  77: 'stair',
  78: 'stool',
  79: 'switch',
  80: 'table',
  81: 'table-runner',
  82: 'tablet',
  83: 'tissue-paper',
  84: 'toilet',
  85: 'toothbrush',
  86: 'towel',
  87: 'tv-screen',
  88: 'tv-stand',
  89: 'umbrella',
  90: 'utensil-holder',
  91: 'vase',
  92: 'vent',
  93: 'wall',
  94: 'wall-cabinet',
  95: 'wall-plug',
  96: 'wardrobe',
  97: 'window',
  98: 'rug',
  99: 'logo',
  100: 'bag',
  101: 'set-of-clothing'
}

AD20K_to_replica_class_ids = {
10: 2, # cabinet -> base-cabinet
112: 3, # basket, handbasket -> basket
37: 4, # tub -> bathtub
115: 6, # bag -> beanbag
7: 7, # bed -> bed
69: 8, # bench -> bench
116: 9, # minibike, motorbike -> bike
10: 10, # cabinet -> bin
131: 11, # blanket, cover -> blanket
63: 12, # blind, screen -> blinds
86: 12, # awning, sunshade, sunblind -> blinds
62: 13, # bookcase -> book
67: 13, # book -> book
98: 14, # bottle -> bottle
41: 15, # box -> box
50: 15, # refrigerator, icebox -> box
10: 18, # cabinet -> cabinet
19: 20, # chair -> chair
30: 20, # armchair -> chair
75: 20, # swivel chair -> chair
148: 22, # clock -> clock
147: 28, # glass, drinking glass -> cup
39: 29, # cushion -> cushion
18: 30, # curtain -> curtain
5: 31, # ceiling -> ceiling
70: 33, # countertop -> countertop
33: 34, # desk -> desk
74: 36, # computer -> desktop-computer
14: 37, # door -> door
58: 37, # screen door, screen -> door
119: 38, # ball -> excercise-ball
3: 40, # floor -> floor
115: 41, # bag -> handbag
95: 43, # bannister, banister, balustrade, balusters, handrail -> handrail
17: 44, # plant -> indoor-plant
36: 47, # lamp -> lamp
87: 47, # street lamp -> lamp
74: 48, # computer -> laptop
107: 50, # washer, automatic washer, washing machine -> mat
124: 51, # microwave -> microwave
143: 52, # monitor -> monitor
61: 55, # bridge, span -> pan
22: 59, # painting, picture -> picture
42: 60, # column, pillar -> pillar
57: 61, # pillow -> pillow
17: 63, # plant -> plant-stand
142: 64, # plate -> plate
65: 65, # toilet, can, commode, crapper, pot, potty, stool, throne -> pot
125: 65, # pot -> pot
91: 66, # dirt track -> rack
50: 67, # refrigerator, icebox -> refrigerator
132: 70, # sculpture -> sculpture
24: 71, # shelf -> shelf
145: 73, # shower -> shower-stall
47: 74, # sink -> sink
23: 76, # sofa -> sofa
53: 77, # stairs -> stair
59: 77, # stairway, staircase -> stair
96: 77, # escalator, moving staircase, moving stairway -> stair
121: 77, # step, stair -> stair
97: 78, # ottoman, pouf, pouffe, puff, hassock -> stool
15: 80, # table -> table
56: 80, # pool table, billiard table, snooker table -> table
64: 80, # coffee table -> table
65: 84, # toilet, can, commode, crapper, pot, potty, stool, throne -> toilet
81: 86, # towel -> towel
89: 87, # tv -> tv-screen
89: 88, # tv -> tv-stand
135: 91, # vase -> vase
0: 93, # wall -> wall
10: 94, # cabinet -> wall-cabinet
35: 96, # wardrobe, closet, press -> wardrobe
8: 97, # window  -> window
28: 98, # rug -> rug
115: 100, # bag -> bag
}