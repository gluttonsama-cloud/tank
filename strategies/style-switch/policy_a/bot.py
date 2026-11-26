"""Grid navigation and short-horizon action selection for the SeedCup arcade arena."""
from collections import deque
from math import ceil, atan2, cos, degrees, hypot, radians, sin

from seedcup import Bot
from seedcup.game import rules
from seedcup.game.bots import clear_shot, wall_grid, ray_wall


ZONE_LOOKAHEAD = 30

# Fraction of constant-velocity displacement fitted on local SDK games.
LEAD = {(0, 1, 0, 0): 0.3881,
 (0, 1, 0, 1): 0.53504,
 (0, 1, 1, 0): 0.25184,
 (0, 1, 1, 1): 0.63609,
 (0, 1, 2, 0): 0.15697,
 (0, 1, 2, 1): 0.62282,
 (0, 1, 3, 0): 0.08562,
 (0, 1, 3, 1): 0.61948,
 (0, 1, 4, 0): 0.04972,
 (0, 1, 4, 1): 0.58425,
 (0, 1, 5, 0): 0.05518,
 (0, 1, 5, 1): 0.55401,
 (0, 1, 6, 0): 0.08039,
 (0, 1, 6, 1): 0.59575,
 (1, 1, 0, 0): 0.22536,
 (1, 1, 0, 1): 0.15322,
 (1, 1, 1, 0): 0.19554,
 (1, 1, 1, 1): 0.25173,
 (1, 1, 2, 0): 0.17055,
 (1, 1, 2, 1): 0.30222,
 (1, 1, 3, 0): 0.14509,
 (1, 1, 3, 1): 0.31744,
 (1, 1, 4, 0): 0.13671,
 (1, 1, 4, 1): 0.32033,
 (1, 1, 5, 0): 0.07563,
 (1, 1, 5, 1): 0.29895,
 (1, 1, 6, 0): 0.08933,
 (1, 1, 6, 1): 0.23964}


MOTION = {(0, 0, 0): [0.05903, -0.00111],
 (0, 0, 0, 0, 0): [0.04458, -0.00152],
 (0, 0, 0, 0, 0, 0, 0): [0.10316, 0.05192],
 (0, 0, 0, 0, 0, 0, 1): [0.09455, -0.07295],
 (0, 0, 0, 0, 0, 1, 0): [0.04202, 0.00734],
 (0, 0, 0, 0, 0, 1, 1): [0.04233, -0.00989],
 (0, 0, 0, 0, 1): [0.07737, -0.00059],
 (0, 0, 0, 0, 1, 1, 0): [0.07668, -0.0053],
 (0, 0, 0, 0, 1, 1, 1): [0.07813, 0.00465],
 (0, 0, 1): [0.13419, -0.00153],
 (0, 0, 1, 0, 0): [0.11822, -0.00181],
 (0, 0, 1, 0, 0, 0, 0): [0.09491, 0.01631],
 (0, 0, 1, 0, 0, 0, 1): [0.09495, -0.01083],
 (0, 0, 1, 0, 0, 1, 0): [0.13548, 0.02105],
 (0, 0, 1, 0, 0, 1, 1): [0.13778, -0.0293],
 (0, 0, 1, 0, 1): [0.23522, 0.00023],
 (0, 0, 1, 0, 1, 1, 0): [0.23251, -0.06624],
 (0, 0, 1, 0, 1, 1, 1): [0.23724, 0.05895],
 (0, 1, 0): [0.04852, -0.00081],
 (0, 1, 0, 0, 0): [0.0354, -0.0006],
 (0, 1, 0, 0, 0, 0, 0): [0.02593, 0.01282],
 (0, 1, 0, 0, 0, 0, 1): [0.02756, -0.01391],
 (0, 1, 0, 0, 0, 1, 0): [0.03726, 0.01303],
 (0, 1, 0, 0, 0, 1, 1): [0.03647, -0.01375],
 (0, 1, 0, 0, 1): [0.06315, -0.00103],
 (0, 1, 0, 0, 1, 0, 0): [0.15247, -0.00747],
 (0, 1, 0, 0, 1, 0, 1): [0.15472, 0.00102],
 (0, 1, 0, 0, 1, 1, 0): [0.06245, -0.01332],
 (0, 1, 0, 0, 1, 1, 1): [0.06144, 0.01231],
 (0, 1, 1): [0.14662, -0.0003],
 (0, 1, 1, 0, 0): [0.08631, 0.00124],
 (0, 1, 1, 0, 0, 0, 0): [0.09968, 0.00588],
 (0, 1, 1, 0, 0, 0, 1): [0.09364, -0.0051],
 (0, 1, 1, 0, 0, 1, 0): [0.07433, 0.00825],
 (0, 1, 1, 0, 0, 1, 1): [0.07153, -0.00575],
 (0, 1, 1, 0, 1): [0.21961, -0.00216],
 (0, 1, 1, 0, 1, 0, 0): [0.23605, -0.00255],
 (0, 1, 1, 0, 1, 0, 1): [0.23669, 0.00309],
 (0, 1, 1, 0, 1, 1, 0): [0.19024, -0.0533],
 (0, 1, 1, 0, 1, 1, 1): [0.1942, 0.05256],
 (0, 2, 0): [0.03069, -2e-05],
 (0, 2, 0, 0, 0): [0.02024, 0.00117],
 (0, 2, 0, 0, 0, 0, 0): [0.01946, 0.00502],
 (0, 2, 0, 0, 0, 0, 1): [0.02238, -0.00352],
 (0, 2, 0, 0, 0, 1, 0): [0.01954, 0.01756],
 (0, 2, 0, 0, 0, 1, 1): [0.02076, -0.01614],
 (0, 2, 0, 0, 1): [0.04265, -0.00136],
 (0, 2, 0, 0, 1, 0, 0): [0.12756, 0.00088],
 (0, 2, 0, 0, 1, 0, 1): [0.13963, -5e-05],
 (0, 2, 0, 0, 1, 1, 0): [0.03969, -0.01738],
 (0, 2, 0, 0, 1, 1, 1): [0.03843, 0.01677],
 (0, 2, 1): [0.13229, -0.00028],
 (0, 2, 1, 0, 0): [0.044, -0.0007],
 (0, 2, 1, 0, 0, 0, 0): [0.04943, -0.00364],
 (0, 2, 1, 0, 0, 0, 1): [0.05594, -0.00162],
 (0, 2, 1, 0, 0, 1, 0): [0.03405, 0.00581],
 (0, 2, 1, 0, 0, 1, 1): [0.03105, -0.00399],
 (0, 2, 1, 0, 1): [0.19095, 0.0],
 (0, 2, 1, 0, 1, 0, 0): [0.20317, -0.00406],
 (0, 2, 1, 0, 1, 0, 1): [0.19687, 0.00357],
 (0, 2, 1, 0, 1, 1, 0): [0.15416, -0.04531],
 (0, 2, 1, 0, 1, 1, 1): [0.15075, 0.04485],
 (0, 3, 0): [0.01886, 0.00173],
 (0, 3, 0, 0, 0): [0.0113, 0.00065],
 (0, 3, 0, 0, 0, 0, 0): [0.01504, 0.00205],
 (0, 3, 0, 0, 0, 0, 1): [0.01668, -0.00464],
 (0, 3, 0, 0, 0, 1, 0): [0.01132, 0.01904],
 (0, 3, 0, 0, 0, 1, 1): [0.01074, -0.01603],
 (0, 3, 0, 0, 1): [0.02605, 0.00276],
 (0, 3, 0, 0, 1, 0, 0): [0.17691, -0.00918],
 (0, 3, 0, 0, 1, 0, 1): [0.15072, 0.00445],
 (0, 3, 0, 0, 1, 1, 0): [0.02303, -0.01671],
 (0, 3, 0, 0, 1, 1, 1): [0.0231, 0.022],
 (0, 3, 1): [0.13597, 0.00027],
 (0, 3, 1, 0, 0): [0.02672, -0.00039],
 (0, 3, 1, 0, 0, 0, 0): [0.03081, -0.00177],
 (0, 3, 1, 0, 0, 0, 1): [0.03592, -0.00077],
 (0, 3, 1, 0, 0, 1, 0): [0.01065, 0.00186],
 (0, 3, 1, 0, 0, 1, 1): [0.01954, -0.00031],
 (0, 3, 1, 0, 1): [0.17725, 0.00052],
 (0, 3, 1, 0, 1, 0, 0): [0.17538, -0.00456],
 (0, 3, 1, 0, 1, 0, 1): [0.18287, 0.00259],
 (0, 3, 1, 0, 1, 1, 0): [0.13167, -0.03388],
 (0, 3, 1, 0, 1, 1, 1): [0.14018, 0.03933],
 (0, 4, 0): [0.00803, -0.00099],
 (0, 4, 0, 0, 0): [0.0038, 0.00024],
 (0, 4, 0, 0, 0, 0, 0): [0.00994, 0.00455],
 (0, 4, 0, 0, 0, 0, 1): [0.00977, 0.00859],
 (0, 4, 0, 0, 0, 1, 0): [0.00348, 0.01352],
 (0, 4, 0, 0, 0, 1, 1): [0.00376, -0.0142],
 (0, 4, 0, 0, 1): [0.01182, -0.00209],
 (0, 4, 0, 0, 1, 0, 0): [0.14869, -0.00358],
 (0, 4, 0, 0, 1, 0, 1): [0.14328, 0.007],
 (0, 4, 0, 0, 1, 1, 0): [0.0111, -0.0196],
 (0, 4, 0, 0, 1, 1, 1): [0.00927, 0.01599],
 (0, 4, 1): [0.14865, -0.00098],
 (0, 4, 1, 0, 0): [0.0126, -0.00051],
 (0, 4, 1, 0, 0, 0, 0): [0.01011, -0.00441],
 (0, 4, 1, 0, 0, 0, 1): [0.01956, 0.00252],
 (0, 4, 1, 0, 0, 1, 0): [0.0055, -0.00191],
 (0, 4, 1, 0, 0, 1, 1): [0.00841, -0.0002],
 (0, 4, 1, 0, 1): [0.18623, -0.00111],
 (0, 4, 1, 0, 1, 0, 0): [0.17293, -0.00698],
 (0, 4, 1, 0, 1, 0, 1): [0.19285, 0.00167],
 (0, 4, 1, 0, 1, 1, 0): [0.16267, -0.052],
 (0, 4, 1, 0, 1, 1, 1): [0.13117, 0.04214],
 (0, 5, 0): [0.00619, 0.00152],
 (0, 5, 0, 0, 0): [0.004, 0.00111],
 (0, 5, 0, 0, 0, 0, 0): [0.00355, -0.00648],
 (0, 5, 0, 0, 0, 0, 1): [0.00941, -0.00651],
 (0, 5, 0, 0, 0, 1, 0): [0.00429, 0.01273],
 (0, 5, 0, 0, 0, 1, 1): [0.00346, -0.01035],
 (0, 5, 0, 0, 1): [0.008, 0.00185],
 (0, 5, 0, 0, 1, 0, 0): [0.11884, -0.00999],
 (0, 5, 0, 0, 1, 0, 1): [0.1332, 0.0056],
 (0, 5, 0, 0, 1, 1, 0): [0.00547, -0.01575],
 (0, 5, 0, 0, 1, 1, 1): [0.00744, 0.01957],
 (0, 5, 1): [0.13193, 0.00049],
 (0, 5, 1, 0, 0): [0.01005, 0.00133],
 (0, 5, 1, 0, 0, 0, 0): [0.00998, 0.00027],
 (0, 5, 1, 0, 0, 0, 1): [0.01374, 0.0031],
 (0, 5, 1, 0, 0, 1, 0): [-0.00533, -0.00542],
 (0, 5, 1, 0, 0, 1, 1): [-0.0009, -0.00071],
 (0, 5, 1, 0, 1): [0.16453, 0.00027],
 (0, 5, 1, 0, 1, 0, 0): [0.1425, -0.00295],
 (0, 5, 1, 0, 1, 0, 1): [0.17135, 0.00086],
 (0, 5, 1, 0, 1, 1, 0): [0.16027, -0.04863],
 (0, 5, 1, 0, 1, 1, 1): [0.14833, 0.04461],
 (0, 6, 0): [0.00777, 0.00099],
 (0, 6, 0, 0, 0): [0.0067, 0.00031],
 (0, 6, 0, 0, 0, 0, 0): [0.00961, -0.00343],
 (0, 6, 0, 0, 0, 0, 1): [0.01113, -0.00235],
 (0, 6, 0, 0, 0, 1, 0): [0.00685, 0.0079],
 (0, 6, 0, 0, 0, 1, 1): [0.00628, -0.00602],
 (0, 6, 0, 0, 1): [0.00858, 0.0015],
 (0, 6, 0, 0, 1, 0, 0): [0.11792, -0.00317],
 (0, 6, 0, 0, 1, 0, 1): [0.13868, 0.00482],
 (0, 6, 0, 0, 1, 1, 0): [0.00447, -0.02196],
 (0, 6, 0, 0, 1, 1, 1): [0.0068, 0.0226],
 (0, 6, 1): [0.13702, -0.00022],
 (0, 6, 1, 0, 0): [0.01221, -0.00212],
 (0, 6, 1, 0, 0, 0, 0): [0.01268, -0.00735],
 (0, 6, 1, 0, 0, 0, 1): [0.01398, 0.00254],
 (0, 6, 1, 0, 0, 1, 0): [-0.00454, -0.00896],
 (0, 6, 1, 0, 0, 1, 1): [0.00626, 0.00175],
 (0, 6, 1, 0, 1): [0.15689, 8e-05],
 (0, 6, 1, 0, 1, 0, 0): [0.12503, -0.00481],
 (0, 6, 1, 0, 1, 0, 1): [0.16721, 0.00158],
 (0, 6, 1, 0, 1, 1, 0): [0.13857, -0.05373],
 (0, 6, 1, 0, 1, 1, 1): [0.11662, 0.0405],
 (0, 7, 0): [0.14744, -0.00132],
 (0, 7, 0, 0, 0): [0.00534, -0.00165],
 (0, 7, 0, 0, 0, 0, 0): [0.02145, -0.01059],
 (0, 7, 0, 0, 0, 0, 1): [0.00775, 0.00575],
 (0, 7, 0, 0, 0, 1, 0): [0.00173, 0.00394],
 (0, 7, 0, 0, 0, 1, 1): [-0.00033, -0.00342],
 (0, 7, 0, 0, 1): [0.16408, -0.00128],
 (0, 7, 0, 0, 1, 0, 0): [0.08489, -0.00078],
 (0, 7, 0, 0, 1, 0, 1): [0.18698, -0.00035],
 (0, 7, 0, 0, 1, 1, 0): [0.01271, -0.04474],
 (0, 7, 0, 0, 1, 1, 1): [-0.00299, 0.03068],
 (0, 7, 1): [0.15723, -0.0004],
 (0, 7, 1, 0, 0): [0.02894, -0.00036],
 (0, 7, 1, 0, 0, 0, 0): [0.02682, 0.0033],
 (0, 7, 1, 0, 0, 0, 1): [0.02915, -0.00072],
 (0, 7, 1, 0, 1): [0.1626, -0.00041],
 (0, 7, 1, 0, 1, 0, 0): [0.15427, -0.00536],
 (0, 7, 1, 0, 1, 0, 1): [0.16499, 0.00101],
 (0, 8, 0): [0.08914, 0.00056],
 (0, 8, 0, 0, 1): [0.08914, 0.00056],
 (0, 8, 0, 0, 1, 0, 0): [0.07905, -0.00296],
 (0, 8, 0, 0, 1, 0, 1): [0.09988, 0.0043],
 (0, 8, 1): [0.07845, 9e-05],
 (0, 8, 1, 0, 1): [0.07845, 9e-05],
 (0, 8, 1, 0, 1, 0, 0): [0.06605, -0.00058],
 (0, 8, 1, 0, 1, 0, 1): [0.09084, 0.00076],
 (1, 0, 0): [0.0, 0.0],
 (1, 0, 0, 0, 0): [0.0, 0.0],
 (1, 0, 0, 0, 0, 0, 0): [0.0, 0.0],
 (1, 0, 0, 0, 0, 0, 1): [0.0, 0.0],
 (1, 0, 0, 0, 0, 1, 0): [0.0, 0.0],
 (1, 0, 0, 0, 0, 1, 1): [0.0, 0.0],
 (1, 0, 0, 0, 1): [0.0, 0.0],
 (1, 0, 0, 0, 1, 1, 0): [0.0, 0.0],
 (1, 0, 0, 0, 1, 1, 1): [0.0, 0.0],
 (1, 0, 1): [0.0, 0.0],
 (1, 0, 1, 0, 0): [0.0, 0.0],
 (1, 0, 1, 0, 0, 0, 0): [0.0, 0.0],
 (1, 0, 1, 0, 0, 0, 1): [0.0, 0.0],
 (1, 0, 1, 0, 0, 1, 0): [0.0, 0.0],
 (1, 0, 1, 0, 0, 1, 1): [0.0, 0.0],
 (1, 0, 1, 0, 1): [0.0, 0.0],
 (1, 0, 1, 0, 1, 1, 0): [0.0, 0.0],
 (1, 0, 1, 0, 1, 1, 1): [0.0, 0.0],
 (1, 1, 0): [0.00802, 2e-05],
 (1, 1, 0, 0, 0): [0.0056, 4e-05],
 (1, 1, 0, 0, 0, 0, 0): [0.00476, 0.00395],
 (1, 1, 0, 0, 0, 0, 1): [0.00438, -0.00407],
 (1, 1, 0, 0, 0, 1, 0): [0.00782, -0.00045],
 (1, 1, 0, 0, 0, 1, 1): [0.00777, -1e-05],
 (1, 1, 0, 0, 1): [0.01166, -0.0],
 (1, 1, 0, 0, 1, 0, 0): [0.0165, -0.00103],
 (1, 1, 0, 0, 1, 0, 1): [0.01282, 0.00035],
 (1, 1, 0, 0, 1, 1, 0): [0.01111, -0.0017],
 (1, 1, 0, 0, 1, 1, 1): [0.01099, 0.00194],
 (1, 1, 1): [0.04115, -0.00128],
 (1, 1, 1, 0, 0): [0.03543, -0.00081],
 (1, 1, 1, 0, 0, 0, 0): [0.03686, 0.0042],
 (1, 1, 1, 0, 0, 0, 1): [0.03086, -0.01191],
 (1, 1, 1, 0, 0, 1, 0): [0.03438, 0.00463],
 (1, 1, 1, 0, 0, 1, 1): [0.04045, -0.00081],
 (1, 1, 1, 0, 1): [0.04611, -0.00168],
 (1, 1, 1, 0, 1, 0, 0): [0.08117, -0.0009],
 (1, 1, 1, 0, 1, 0, 1): [0.09348, -0.00235],
 (1, 1, 1, 0, 1, 1, 0): [0.04017, -0.012],
 (1, 1, 1, 0, 1, 1, 1): [0.03537, 0.01184],
 (1, 2, 0): [0.00341, 7e-05],
 (1, 2, 0, 0, 0): [0.00201, 0.0001],
 (1, 2, 0, 0, 0, 0, 0): [0.00186, 0.00039],
 (1, 2, 0, 0, 0, 0, 1): [0.00199, -0.00018],
 (1, 2, 0, 0, 0, 1, 0): [0.00332, -0.00013],
 (1, 2, 0, 0, 0, 1, 1): [0.00272, 6e-05],
 (1, 2, 0, 0, 1): [0.00952, -6e-05],
 (1, 2, 0, 0, 1, 0, 0): [0.0128, -4e-05],
 (1, 2, 0, 0, 1, 0, 1): [0.01177, 0.00019],
 (1, 2, 0, 0, 1, 1, 0): [0.00602, -0.0013],
 (1, 2, 0, 0, 1, 1, 1): [0.00525, 0.00099],
 (1, 2, 1): [0.03726, -0.00097],
 (1, 2, 1, 0, 0): [0.02048, 0.00078],
 (1, 2, 1, 0, 0, 0, 0): [0.02119, -0.00056],
 (1, 2, 1, 0, 0, 0, 1): [0.01977, -0.00106],
 (1, 2, 1, 0, 1): [0.04755, -0.00205],
 (1, 2, 1, 0, 1, 0, 0): [0.07219, -0.00047],
 (1, 2, 1, 0, 1, 0, 1): [0.04829, -0.00098],
 (1, 2, 1, 0, 1, 1, 0): [0.03913, -0.00888],
 (1, 2, 1, 0, 1, 1, 1): [0.01224, 0.00301],
 (1, 3, 0): [0.00399, -0.00016],
 (1, 3, 0, 0, 0): [0.00102, -0.00017],
 (1, 3, 0, 0, 0, 0, 0): [0.00087, -9e-05],
 (1, 3, 0, 0, 0, 0, 1): [0.00116, -0.00026],
 (1, 3, 0, 0, 0, 1, 0): [0.00071, -6e-05],
 (1, 3, 0, 0, 0, 1, 1): [0.00162, 0.00027],
 (1, 3, 0, 0, 1): [0.01386, -0.00015],
 (1, 3, 0, 0, 1, 0, 0): [0.01648, -0.00068],
 (1, 3, 0, 0, 1, 0, 1): [0.0156, 0.00043],
 (1, 3, 0, 0, 1, 1, 0): [0.00276, -0.00065],
 (1, 3, 0, 0, 1, 1, 1): [0.00186, 0.00011],
 (1, 3, 1): [0.00082, -0.00018],
 (1, 3, 1, 0, 1): [0.0, 0.0],
 (1, 3, 1, 0, 1, 0, 0): [0.0, 0.0],
 (1, 3, 1, 0, 1, 0, 1): [0.0, 0.0],
 (1, 4, 0): [0.00393, 0.00016],
 (1, 4, 0, 0, 0): [0.00092, 0.00017],
 (1, 4, 0, 0, 0, 0, 0): [0.00102, 7e-05],
 (1, 4, 0, 0, 0, 0, 1): [0.00088, 0.00024],
 (1, 4, 0, 0, 0, 1, 0): [0.00065, 0.00039],
 (1, 4, 0, 0, 0, 1, 1): [-7e-05, 0.00057],
 (1, 4, 0, 0, 1): [0.01145, 0.00012],
 (1, 4, 0, 0, 1, 0, 0): [0.01273, -0.00024],
 (1, 4, 0, 0, 1, 0, 1): [0.01458, 0.00045],
 (1, 4, 0, 0, 1, 1, 0): [0.00079, -2e-05],
 (1, 4, 0, 0, 1, 1, 1): [0.00062, 0.00016],
 (1, 4, 1): [0.0, 0.0],
 (1, 4, 1, 0, 1): [0.0, 0.0],
 (1, 4, 1, 0, 1, 0, 0): [0.0, 0.0],
 (1, 4, 1, 0, 1, 0, 1): [0.0, 0.0],
 (1, 5, 0): [0.00444, -0.00031],
 (1, 5, 0, 0, 0): [0.00082, -0.00031],
 (1, 5, 0, 0, 0, 0, 0): [0.00074, -0.00027],
 (1, 5, 0, 0, 0, 0, 1): [0.0009, -0.00037],
 (1, 5, 0, 0, 0, 1, 0): [0.00029, 9e-05],
 (1, 5, 0, 0, 0, 1, 1): [0.00128, -0.00029],
 (1, 5, 0, 0, 1): [0.01262, -0.00033],
 (1, 5, 0, 0, 1, 0, 0): [0.01572, -0.00102],
 (1, 5, 0, 0, 1, 0, 1): [0.0141, 0.00031],
 (1, 5, 0, 0, 1, 1, 0): [0.00062, -0.00011],
 (1, 5, 0, 0, 1, 1, 1): [0.00036, -2e-05],
 (1, 5, 1): [0.0038, -6e-05],
 (1, 5, 1, 0, 1): [0.00407, 0.0],
 (1, 5, 1, 0, 1, 0, 0): [0.0, 0.0],
 (1, 5, 1, 0, 1, 0, 1): [0.01105, 0.0],
 (1, 6, 0): [0.00703, -6e-05],
 (1, 6, 0, 0, 0): [0.00093, -0.0002],
 (1, 6, 0, 0, 0, 0, 0): [0.00092, -0.00027],
 (1, 6, 0, 0, 0, 0, 1): [0.00092, -0.00012],
 (1, 6, 0, 0, 0, 1, 0): [0.00122, -0.0008],
 (1, 6, 0, 0, 0, 1, 1): [0.00094, 2e-05],
 (1, 6, 0, 0, 1): [0.0164, 0.00016],
 (1, 6, 0, 0, 1, 0, 0): [0.01694, -0.00048],
 (1, 6, 0, 0, 1, 0, 1): [0.01949, 0.00089],
 (1, 6, 0, 0, 1, 1, 0): [0.0005, 3e-05],
 (1, 6, 0, 0, 1, 1, 1): [0.00028, -3e-05],
 (1, 6, 1): [0.00841, 0.00016],
 (1, 6, 1, 0, 1): [0.00898, 0.00012],
 (1, 6, 1, 0, 1, 0, 0): [0.00815, -2e-05],
 (1, 6, 1, 0, 1, 0, 1): [0.01152, 0.00029],
 (1, 7, 0): [0.00365, -1e-05],
 (1, 7, 0, 0, 0): [0.00191, -7e-05],
 (1, 7, 0, 0, 0, 0, 0): [0.00209, -0.00027],
 (1, 7, 0, 0, 0, 0, 1): [0.00182, 0.0001],
 (1, 7, 0, 0, 0, 1, 0): [0.00055, 0.00087],
 (1, 7, 0, 0, 0, 1, 1): [0.0, 0.0],
 (1, 7, 0, 0, 1): [0.00383, 0.0],
 (1, 7, 0, 0, 1, 0, 0): [0.00476, -0.00011],
 (1, 7, 0, 0, 1, 0, 1): [0.00307, 0.0001],
 (1, 7, 0, 0, 1, 1, 0): [0.0, 0.0],
 (1, 7, 0, 0, 1, 1, 1): [0.0, 0.0],
 (1, 7, 1): [0.0136, -2e-05],
 (1, 7, 1, 0, 1): [0.01439, -5e-05],
 (1, 7, 1, 0, 1, 0, 0): [0.00902, -7e-05],
 (1, 7, 1, 0, 1, 0, 1): [0.02172, -2e-05],
 (1, 8, 0): [0.00484, -2e-05],
 (1, 8, 0, 0, 1): [0.00484, -2e-05],
 (1, 8, 0, 0, 1, 0, 0): [0.00531, -0.0002],
 (1, 8, 0, 0, 1, 0, 1): [0.00437, 0.00016]}

def motion_features(bot,x,y,ex,ey,eh,vx,vy,previous_heading):
    dist=hypot(ex-x,ey-y)
    ticks=max(1,min(24,ceil((dist-.92)/1.2)))
    heading=radians(eh)
    dx,dy=cos(heading),sin(heading)
    speed=vx*dx+vy*dy
    seen=int(clear_shot(bot.walls,ex,ey,x,y))
    turning=(eh-previous_heading+180)%360-180
    keys=(seen,round(ticks/3),int(speed>.1),int(speed<-.06),
          int(bot.free(ex+.75*dx,ey+.75*dy)),int(abs(turning)>6),
          int((dx*(y-ey)-dy*(x-ex))>=0))
    return keys,ticks,dx,dy

def angle(dx, dy):
    return degrees(atan2(dy, dx))


def delta(target, current):
    return (target - current + 180) % 360 - 180


def clamp(value, limit):
    return max(-limit, min(limit, value))


class Lancer(Bot):
    def on_match_start(self, info):
        self.grid = info["map"]["grid"]
        self.walls = wall_grid(info)
        self.open = {(x, y) for y, row in enumerate(self.grid)
                     for x, wall in enumerate(row) if not wall}
        self.links = {p: [q for q in ((p[0]+1,p[1]), (p[0]-1,p[1]),
                                      (p[0],p[1]+1), (p[0],p[1]-1)) if q in self.open]
                      for p in self.open}
        self.previous = None
        self.route_goal = None
        self.distances = {}
        self.retreating = False

    def waypoint(self, x, y, tx, ty):
        goal = int(tx), int(ty)
        if goal not in self.open:
            goal = min(self.open, key=lambda p: (p[0]+0.5-tx)**2 + (p[1]+0.5-ty)**2)
        if goal != self.route_goal:
            queue = deque([goal])
            dist = {goal: 0}
            while queue:
                p = queue.popleft()
                for q in self.links[p]:
                    if q not in dist:
                        dist[q] = dist[p] + 1
                        queue.append(q)
            self.distances, self.route_goal = dist, goal
        p = int(x), int(y)
        if p == goal:
            return goal[0]+0.5, goal[1]+0.5
        q = min(self.links[p], key=self.distances.__getitem__)
        # In a one-cell corridor, line up with its centre before turning a corner.
        best=q
        for _ in range(6):
            if q==goal:break
            q=min(self.links[q],key=self.distances.__getitem__)
            dx,dy=q[0]+.5-x,q[1]+.5-y
            steps=max(1,ceil(hypot(dx,dy)*4))
            if not all(self.free(x+dx*i/steps,y+dy*i/steps) for i in range(1,steps+1)):break
            best=q
        return best[0]+.5,best[1]+.5

    def free(self, x, y):
        for cx in (int(x-.38), int(x+.38)):
            for cy in (int(y-.38), int(y+.38)):
                if (cx, cy) not in self.open:
                    dx = max(cx-x, 0, x-cx-1)
                    dy = max(cy-y, 0, y-cy-1)
                    if dx*dx + dy*dy < .38**2:
                        return False
        return True

    def evade(self, x, y, heading, preferred, speed, shells, safe_center, frame):
        # Rank short movement arcs by clearance from the visible game projectiles.
        best = None
        cx, cy = safe_center
        radii = [rules.zone_radius(frame + tick) for tick in range(1, 9)]
        already_outside = hypot(x-cx, y-cy) > radii[0]
        future_radius = rules.zone_radius(frame + ZONE_LOOKAHEAD)
        for turn in (-1, -.5, 0, .5, 1, preferred):
            for forward in (-1, 1, 0):
                px, py, h = x, y, heading
                clearance = 3
                zone_cost = 0
                for tick, radius in enumerate(radii, 1):
                    h += turn*18
                    nx = px + forward*.25*cos(radians(h))
                    ny = py + forward*.25*sin(radians(h))
                    if self.free(nx, ny):
                        px, py = nx, ny
                    if hypot(px-cx, py-cy) > radius:
                        # Prefer an in-zone next step if one is possible.
                        zone_cost += 26 if tick == 1 and not already_outside else .8
                    for sx, sy, vx, vy in shells:
                        along = clamp((px-sx)*vx + (py-sy)*vy - (tick-.5)*1.2, .6)
                        t = (tick-.5)*1.2 + along
                        clearance = min(clearance, hypot(px-sx-vx*t, py-sy-vy*t))
                utility = (min(clearance, .95)*20 - abs(turn-preferred)*.24
                           - abs(forward-speed)*.2
                           - zone_cost - max(0, hypot(px-cx, py-cy)-future_radius)*2)
                if best is None or utility > best[0]:
                    best = utility, turn, forward
        return best[1:]


    def firing_lane(self,x,y,ex,ey):
        start=int(x),int(y)
        queue=deque([(start,())])
        visited={start}
        while queue:
            cell,path=queue.popleft()
            if path and clear_shot(self.walls,cell[0]+.5,cell[1]+.5,ex,ey):
                for point in reversed(path[:3]):
                    dx,dy=point[0]+.5-x,point[1]+.5-y
                    steps=max(1,ceil(hypot(dx,dy)*4))
                    if all(self.free(x+dx*i/steps,y+dy*i/steps) for i in range(1,steps+1)):
                        return point[0]+.5,point[1]+.5
                return path[0][0]+.5,path[0][1]+.5
            if len(path)<6:
                for q in self.links[cell]:
                    if q not in visited:
                        visited.add(q)
                        queue.append((q,path+(q,)))
        return self.waypoint(x,y,ex,ey)

    def dodge(self,x,y,h,preferred,speed,shells,bearing,ready,visible):
        # Predict exact swept segments, truncated where they meet the map.
        paths=[]
        horizon=6
        for sx,sy,dx,dy in shells:
            end=ray_wall(self.walls,sx,sy,dx,dy,horizon*1.2)
            for i in range(horizon):
                length=min(1.2,end-i*1.2)
                if length<=0: break
                if len(paths)<=i:paths.append([])
                paths[i].append((sx+i*1.2*dx,sy+i*1.2*dy,dx,dy,length))
        if not paths:return preferred,speed
        best=None
        for turn in (preferred,-1,-.5,0,.5,1):
            for forward in (speed,1,-1,0):
                px,py,heading=x,y,h
                danger=0
                for i,segments in enumerate(paths):
                    if i < 3:heading+=turn*18
                    nx=px+forward*.25*cos(radians(heading))
                    ny=py+forward*.25*sin(radians(heading))
                    if self.free(nx,ny):px,py=nx,ny
                    elif self.free(nx,py):px=nx
                    elif self.free(px,ny):py=ny
                    for sx,sy,dx,dy,length in segments:
                        t=max(0,min(length,(px-sx)*dx+(py-sy)*dy))
                        distance=hypot(px-sx-t*dx,py-sy-t*dy)
                        danger=max(danger,max(0,.64-distance)*(1-.022*i))
                utility=-120*danger-abs(turn-preferred)*.28-abs(forward-speed)*.12
                if visible:
                    ex,ey=self.previous
                    if not clear_shot(self.walls,px,py,ex,ey):
                        utility-=0.2
                if visible and ready and abs(delta(bearing,h+turn*18))<1:
                    utility+=.4
                if best is None or utility>best[0]:best=utility,turn,forward
        return best[1:]

    def connects(self,x,y,heading,turn,speed,tx0,ty0):
        direction=radians(heading+turn*18)
        dx,dy=cos(direction),sin(direction)
        nx,ny=x+speed*.25*dx,y+speed*.25*dy
        if not self.free(nx,ny):
            if self.free(nx,y):ny=y
            elif self.free(x,ny):nx=x
            else:nx,ny=x,y
        sx,sy=nx+.5*dx,ny+.5*dy
        tx,ty=tx0-sx,ty0-sy
        along=tx*dx+ty*dy
        side=abs(tx*dy-ty*dx)
        if along>0 and side<0.41:
            distance=max(.2,ceil((along-max(0,.42**2-side**2)**.5)/.2)*.2)
            return ray_wall(self.walls,sx,sy,dx,dy,distance)>=distance
        return False

    def decide(self, obs):
        me, enemy = obs["self"], obs["enemy"]
        x, y, heading = me["x"], me["y"], me["heading"]
        ex, ey = enemy["x"], enemy["y"]
        vx, vy = (0, 0) if self.previous is None else (ex-self.previous[0], ey-self.previous[1])
        self.previous = ex, ey
        dist = hypot(ex-x, ey-y)
        flight = max(1, min(24, ceil((dist-.92)/1.2)))
        if hypot(vx, vy) > .08:
            key = (int(clear_shot(self.walls, x, y, ex, ey)), 1, min(6, flight//3),
                   int(self.free(ex+vx*3, ey+vy*3)))
            flight *= LEAD.get(key, 1)
        ax, ay = ex+vx*flight, ey+vy*flight
        eh=enemy["heading"]
        k,t,fx,fy=motion_features(self,x,y,ex,ey,eh,vx,vy,getattr(self,"prior_heading",eh))
        self.prior_heading=eh
        prediction = MOTION.get(k) or MOTION.get(k[:5]) or MOTION.get(k[:3])
        if prediction:
            along, across = prediction
            ax = ex + t * (along * fx - across * fy)
            ay = ey + t * (along * fy + across * fx)
        if not self.free(ax, ay):
            ax, ay = ex, ey
        bearing = angle(ax-x, ay-y)
        visible = clear_shot(self.walls, x, y, ax, ay)
        expected_x,expected_y=ax,ay
        center_visible = visible
        if not visible and dist > 1:
            nx, ny = -(ey-y)/dist, (ex-x)/dist
            for offset in (0.36,-0.36):
                px, py = ax+offset*nx, ay+offset*ny
                if clear_shot(self.walls,x,y,px,py):
                    ax, ay = px, py
                    bearing = angle(ax-x, ay-y)
                    visible = True
                    break
        zone = obs["zone"]
        safe_center = zone["x"], zone["y"]
        safe_radius = rules.zone_radius(obs["frame"] + ZONE_LOOKAHEAD)
        margin = 1.1 if self.retreating else .6
        self.retreating = hypot(x-safe_center[0], y-safe_center[1]) > safe_radius-margin
        cooldown = me["reload"]
        if self.retreating or not visible:
            tx, ty = safe_center if self.retreating else (ex, ey)
            wx, wy = self.waypoint(x,y,tx,ty) if self.retreating else self.firing_lane(x,y,ex,ey)
            target = angle(wx-x, wy-y)
            speed = min(1, hypot(wx-x,wy-y)*4)
        elif not center_visible and dist>6 and cooldown>.15:
            # Move to a full firing lane between shots when only an edge is visible.
            wx,wy=self.firing_lane(x,y,ex,ey)
            target=angle(wx-x,wy-y)
            speed=min(0.65,hypot(wx-x,wy-y)*4)
        elif cooldown <= 0.35 or not center_visible or dist > 7:
            target, speed = bearing, 0
            if center_visible and dist>13 and self.free(x+.5*(ex-x)/dist,y+.5*(ey-y)/dist):
                speed=0.75
        else:
            # Search for an exit around the nearest corner when boxed in.
            cell=(int(x),int(y))
            queue=deque([(cell,None,0)])
            visited={cell}
            best=None
            while queue:
                pos,first,depth=queue.popleft()
                if first:
                    px,py=pos[0]+.5,pos[1]+.5
                    quality=hypot(px-ex,py-ey)-depth*.35
                    if not clear_shot(self.walls,px,py,ex,ey):quality+=1.4
                    if best is None or quality>best[0]:best=quality,first
                if depth<4:
                    for q in self.links[pos]:
                        if q not in visited:
                            visited.add(q)
                            queue.append((q,first or q,depth+1))
            wx,wy=best[1]
            target,speed=angle(wx+.5-x,wy+.5-y),1
        diff = delta(target, heading)
        if speed and abs(diff) > 90:
            diff = delta(target+180, heading)
            speed = -speed
        turn = clamp(diff/18, 1)
        if abs(diff) > 55:
            speed *= .35
        incoming = []
        frame = obs["frame"]
        if any(not sh["mine"] and hypot(sh["x"]-ex,sh["y"]-ey)<1.73 for sh in obs["shells"]):
            self.last_enemy_shot = frame-1
        until = max(0, 15 - (frame-getattr(self,"last_enemy_shot",-100)))
        if dist < 5 and until <= 3 and visible:
            enemy_angle=radians(enemy["heading"])
            dx,dy=cos(enemy_angle),sin(enemy_angle)
            if abs(delta(angle(x-ex,y-ey),enemy["heading"])) < 12:
                # A projected round reaches the muzzle after the opponent reloads.
                offset=.5-until*1.2
                incoming.append((ex+offset*dx,ey+offset*dy,dx,dy))

        for s in obs["shells"]:
            if s["mine"]:
                continue
            dx, dy = x-s["x"], y-s["y"]
            along = dx*s["dx"]+dy*s["dy"]
            if 0 < along < 8.2 and abs(dx*s["dy"]-dy*s["dx"]) < 1.4:
                incoming.append((s["x"],s["y"],s["dx"],s["dy"]))
        if incoming:
            turn, speed = self.dodge(x,y,heading,turn,speed,incoming,bearing,cooldown<=.1,visible) if not self.retreating else self.evade(x,y,heading,turn,speed,incoming,safe_center,obs["frame"])
        if dist < 2.5 and cooldown <= .2 and clear_shot(self.walls, x, y, ex, ey):
            # At contact distance the projectile can hit before either vehicle
            # clears the corner. Finish turning instead of repeatedly evading.
            turn=clamp(delta(angle(ex-x,ey-y),heading)/18,1)
            speed=0
            fire=cooldown<=.1 and self.connects(x,y,heading,turn,0,ex,ey)
            return {"forward":speed,"turn":turn,"fire":fire}
        fire=cooldown<=.1 and visible and self.connects(x,y,heading,turn,speed,expected_x,expected_y)
        if cooldown<=.1 and not fire and not incoming and not self.retreating and dist>0:
            direct=angle(expected_x-x,expected_y-y)
            if abs(delta(direct,heading)) < 23:
                for offset in (0, 0.2, -0.2):
                    aim=direct+offset/max(dist,1)*57.2958
                    adjustment=clamp(delta(aim,heading)/18,1)
                    for advance in (speed,0,1,-1):
                        if self.connects(x,y,heading,adjustment,advance,expected_x,expected_y):
                            return {"forward":advance,"turn":adjustment,"fire":True}
        return {"forward": speed, "turn": turn, "fire": fire}

