from matplotlib import pyplot as plot

class Visualiser:
    def __init__(self):
        pass


    @staticmethod
    def visualise_data(components, y):
        plot.figure()
        plot.title("Data")
        plot.scatter(components[:, 0], components[:, 1], c=y, cmap=plot.cm.coolwarm)
        plot.show()