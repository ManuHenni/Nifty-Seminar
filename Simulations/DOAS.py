import numpy as np
class DOAS:
    """
    ID: assigned ID, should be different for every DOAS class of cause
    position: Position of the DOAS (Array with X, Y, Z)
    orientation: Viewing direction (Array with 3 Angles alpha, beta, gamma)
    """

    def __init__(self, ID, position, orientation):
        self.position = np.array(position)
        self.ID = ID
        self.orientation = np.array(orientation)

    def print_parameters(self):
        print("Type:", "DOAS-Device")
        print("ID:", self.ID)
        print("orientation:", self.orientation)
        print("position:", self.position)
        print("-------------------------------------")

    def update(self, which, *args):
        new_value = args[0]
        if which == "position":
            self.position = new_value
        elif which == "ID":
            self.ID = new_value
        elif which == "type":
            self.orientation = new_value
        else:
            print("Unknown Key.")
            raise KeyError
