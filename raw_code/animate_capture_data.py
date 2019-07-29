import numpy as np
import pickle
import csv
import sys


MOTION_CAPTURE_FILENAME = sys.argv[1]
# RADAR_DATA_FILENAME = sys.argv[2]
# PICKLE_DATA_FILENAME = sys.argv[2]

NAME_OF_OBJ = "radar_group6"

# print(MOTION_CAPTURE_FILENAME)
# print(RADAR_DATA_FILENAME)

ALL_INFO_INDS = []
POS_INFO_INDS = []

POSITIONS = []

#PARSING POSITION DATA FOR MOCAP
with open(MOTION_CAPTURE_FILENAME, newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='|')
    r_ind = 0
    for row in reader :
        # print("next rows")
        for i in range(len(row)) :
            # print(i)
            if (r_ind == 3) and (row[i] == NAME_OF_OBJ) :
                ALL_INFO_INDS.append(i)
            if (r_ind == 5) and (i in ALL_INFO_INDS) and (row[i] == "Position") :
                POS_INFO_INDS.append(i)
        if r_ind >= 7 :
            if row[POS_INFO_INDS[0]] != "" :
                # POSITIONS.append( [[float(row[POS_INFO_INDS[0]]), float(row[POS_INFO_INDS[1]]), float(row[POS_INFO_INDS[2]])], [float(row[1])]] )
                POSITIONS.append( [float(row[POS_INFO_INDS[0]]), float(row[POS_INFO_INDS[1]]), float(row[POS_INFO_INDS[2]])] )

        r_ind += 1
        # input()
        # print(POSITIONS)


#ANIMATE MOVEMENT DATA

# from matplotlib import pyplot as plt
# from matplotlib import animation

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(-20, 20), ylim=(-20, 20))
# line, = ax.plot([], [], lw=2)

# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,

# # animation function.  This is called sequentially
# def animate(i):
#     if POSITIONS[i][0][0] != "" :
#         x = [float(POSITIONS[i][0][0]), float(POSITIONS[i][0][0])+.1]
#         y = [float(POSITIONS[i][0][1]), float(POSITIONS[i][0][1])+.1]
#         line.set_data(x, y)
#     return line,

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=len(POSITIONS), interval=1, blit=True)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
# from IPython.display import HTML

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        line.set_marker(".")
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Lines to plot in 3D
t = np.array([i for i in range(len(POSITIONS))])

# print(np.array(POSITIONS[:][0][1]))


POSITIONS = np.array(POSITIONS)

# print(POSITIONS.shape)
# print(POSITIONS)
# # print(POSITIONS[:,0])

# print(POSITIONS[:,0][:,0])

x1, y1, z1 = np.array(POSITIONS[:,0]), np.array(POSITIONS[:,1]), np.array(POSITIONS[:,2])
x2, y2, z2 = np.array(POSITIONS[:,0]), np.array(POSITIONS[:,1]), np.array(POSITIONS[:,2])
data = np.array([[x1,y1,z1],[x2,y2,z2]])

print(data.shape)
print(data)

# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

ax.set_xlim(-6,-4)
ax.set_ylim(1,2)
ax.set_zlim(-3,1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
# plt.rcParams['animation.html'] = 'html5'
line_ani = animation.FuncAnimation(fig, update_lines, len(POSITIONS), fargs=(data, lines),
                                   interval=1, blit=True, repeat=True)

# line_ani.save('line_animation_3d_funcanimation.mp4', writer='ffmpeg',fps=1000/100)
plt.show()