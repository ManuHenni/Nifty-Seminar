import numpy as np
class RetroReflector:
    """
    ID: assigned ID, should be different for every Retro Reflector class of cause
    position: Position of the Retro Reflector (Array with X, Y, Z)
    """

    def __init__(self, ID, position):
        self.position = np.array(position)
        self.ID = ID

    def print_parameters(self):
        print("Type:", "Retro-Reflector")
        print("ID:", self.ID)
        print("position:", self.position)
        print("-------------------------------------")


