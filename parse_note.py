import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter, Locator, NullLocator
import os
import glob

class myLocator(Locator):
    def __init__(self, scale=1):
        super(myLocator, self).__init__()
        self.scale=scale
    def __call__(self):
        return self.tick_values(None,None)
    def tick_values(self, vmin, vmax):
        return np.arange(0.5, 88, self.scale)

def funcx(x, pos):
    return int(x/50)
def funcy(x, pos):
    # print(x, pos)
    return int(x+20.5)
    
class Parse_note:
    def __init__(self, dir, end):
        self.dir = dir 
        self.end = end
    
    def get_note(self):
        with open(self.dir) as f:
            lines = f.readlines()
            lines.insert(0, '0,\n') # add start

        notes = []
        for line in lines:
            line_time = int(line.split(',')[0])
            if len(line.split(',')[1])==1: 
                line_note = []
            else:
                line_note = [int(i) for i in line.split(',')[1][:-2].split(' ')]
                # line_note = [int(i) for i in line.split(',')[1].split(' ')]

            notes.append((line_time, line_note))
            # print(line_time, line_note, type(line_time), type(line_note))
            # print('=========================================================')
        # print(notes)

        ajust_notes = []                            # padding note to fill unpredicted time
        for index, note in enumerate(notes):
            ajust_notes.append(note)
            time = 1
            if index==len(notes)-1:     # add end
                for i in range(self.end-index):
                    ajust_notes.append((note[0]+time, note[1]))
                    time = time+1
                break
            time = 1
            while (note[0]+time)<notes[index+1][0]:
                ajust_notes.append((note[0]+time, note[1]))
                time = time+1
        # print(len(ajust_notes))
        # print(ajust_notes)

        plot_notes = np.zeros((len(ajust_notes), 88))  # note range 50~80||convert note to 0-1 array
        for note in ajust_notes:
            for j in note[1]: plot_notes[note[0], j-21] = 1
        # print(plot_notes)
        return plot_notes

    def plot(self):
        plot_notes = self.get_note()
        fig = plt.figure(figsize=(30, 20), dpi=100)
        plt.pcolormesh(plot_notes.T, cmap='jet')
        xLocator = MultipleLocator(50)
        # yLocator = MultipleLocator(6)
        yLocator = myLocator(6)
        # yminLocator = MultipleLocator(1)
        yminLocator = myLocator(1)

        plt.gca().xaxis.set_major_locator(xLocator)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(funcx))

        plt.gca().yaxis.set_major_formatter(FuncFormatter(funcy))
        plt.gca().yaxis.set_major_locator(yLocator)
        plt.gca().yaxis.set_minor_locator(yminLocator)

        plt.grid(color='g', linestyle='--', linewidth=0.5)
        plt.show()

if __name__=='__main__':
    # dir = 'txt/wave_test_0.9_0.2_max.txt'
    # dir = 'txt/wave_test_0.9_0.51_max.txt'
    # dir = 'txt/wave_test_0.9_0.6_no_max.txt'
    # dir = 'txt/wave_test_0.9_0.7_max.txt'
    # dir = 'txt/wave_test_0.9_0.6_no_max_all.txt'
    # dir = 'txt/wave_test_0.9_0.8_no_max_all.txt'
    # dir = 'txt/wave_test_0.9_0.9_no_max_all.txt'
    dir = 'txt/wave_test_0.9_0.99_no_max_all.txt'
    plot = Parse_note(dir, 20)
    plot.plot()