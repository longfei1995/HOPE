import numpy as np
# from shapely.geometry import Point, LinearRing
# from shapely.affinity import affine_transform

# from configs import *
# from env.vehicle import State
# import matplotlib.pyplot as plt

class PathPoint:
    def __init__(self):
        self.x: float = 0
        self.y: float = 0
        self.heading: float = 0
        self.speed: float = 0
        self.steering: float = 0



if __name__ == '__main__':
    scene_types = {0:'Normal', 
                            1:'Complex',
                            2:'Extrem',
                            3:'dlp',
                            }
    print(len(scene_types))
    print(np.zeros(len(scene_types)))
        
    
