import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
import dash
import dash_core_components as dcc
import dash_html_components as html
import math as m
from RetroRefl import RetroReflector
from DOAS import *
from Line import *
from Measurement_Devices import *
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")

def generate_random_field(x_len, y_len, z_len, bubble_amount=15, bubble_scale=4):
    # Generate nicely looking random 3D-field
    np.random.seed(np.random.randint(0,1000))
    X, Y, Z = np.mgrid[:x_len, :y_len, :z_len]
    vol = np.zeros((x_len, y_len, z_len))
    pts = (np.array([x_len * np.random.rand(1, bubble_amount),
                     y_len * np.random.rand(1, bubble_amount),
                     z_len * np.random.rand(1, bubble_amount)])).astype(np.int)
    vol[tuple(indices for indices in pts)] = 1

    vol = ndimage.gaussian_filter(vol, bubble_scale)
    vol /= vol.max()
    return vol, X, Y, Z

def generate_centered_gauss(x_len, y_len, z_len, bubble_amount=1, bubble_scale=3):
    # Generate nicely looking random 3D-field
    X, Y, Z = np.mgrid[:x_len, :y_len, :z_len]
    vol = np.zeros((x_len, y_len, z_len))
    pts = (np.array([x_len,
                     y_len,
                     z_len])).astype(np.int)
    vol[(int(x_len/2), int(y_len/2), int(z_len/2))] = 1

    vol = ndimage.gaussian_filter(vol, bubble_scale)
    vol /= vol.max()
    return vol, X, Y, Z

vol, X_field, Y_field, Z_field = generate_random_field(30,30,21)
#vol, X_field, Y_field, Z_field = generate_centered_gauss(30,30,21)
dimension = "2D"
ground_truth_field = np.sum(vol, axis=2)
normalized_ground_truth_field = (np.sum(vol, axis=2) / np.max(np.sum(vol, axis=2)))
plt.imshow(normalized_ground_truth_field, cmap="inferno")
plt.show()

#create some DOAS devices:
doas1 = DOAS(1, [0,10,10], [1,4,20])
doas2 = DOAS(2, [0,20,10], [1,4,200])
#doas3 = DOAS(3, [-10,29,2], [10,40,200])
doas3 = DOAS(4, [10,0,10], [10,40,200])
doas4 = DOAS(5, [20,0,10], [10,40,200])
DOASs = [doas1, doas2, doas3, doas4]


#create some Reflectors:
refl1 = RetroReflector(1, [30,30,10])
refl2 = RetroReflector(2, [20,30,20])
refl3 = RetroReflector(3, [10,30,10])
refl4 = RetroReflector(4, [30,20,20])
refl5 = RetroReflector(5, [30,10,2])
REFLs = [refl1, refl2, refl3, refl4, refl5]


Measurement_devices = Measurement_Devices(DOASs, REFLs, vol)
lines = Measurement_devices.return_plottables()
DOAS_positions, REFL_positions = Measurement_devices.return_positions()
measurements = Measurement_devices.measure()





#plotting
plot_data = [go.Volume(
    x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
    value=vol.flatten(),
    name=r'NOx-Cloud',
    isomin=0.1,
    isomax=1.,
    opacity=0.1,
    surface_count=25,
    )]

i=0
doas_idx = 0
reflector_idx = 0
doas_ids = [DOAS_device.ID for DOAS_device in DOASs]
reflector_ids = [refl.ID for refl in REFLs]
for X,Y,Z in lines:
    plot_data.append(go.Scatter3d(x=[X[0], X[int(len(X)/8)], X[-1]],
                                  y=[Y[0], Y[int(len(Y)/8)], Y[-1]],
                                  z=[Z[0], Z[int(len(Z)/8)], Z[-1]],
                                  mode='lines+text',
                                  text=["", "{}".format(round(measurements[i], 2))],textfont={"size":14},
                                  name="D{}->R{}".format(doas_ids[doas_idx], reflector_ids[reflector_idx]),
                                  marker=dict(color="rgba(0, 0, 0, 1)")))
    reflector_idx += 1
    if reflector_idx == len(reflector_ids):
        reflector_idx = 0
        doas_idx += 1
    i+=1

i=0
for DOAS_position in DOAS_positions:
    plot_data.append(go.Scatter3d(x=[DOAS_position[0]], y=[DOAS_position[1]], z=[DOAS_position[2]],
                                   mode='markers+text', text="DOAS {}".format(DOASs[i].ID),textfont={"size":18},
                                   marker=dict(size=12.,color="rgba(0, 0, 0, 1)", symbol="cross")))
    i+=1

i=0
for REFL_position in REFL_positions:
    plot_data.append(go.Scatter3d(x=[REFL_position[0]], y=[REFL_position[1]], z=[REFL_position[2]],
                                  mode='markers+text', text="REFL {}".format(REFLs[i].ID),textfont={"size":18},
                                  marker=dict(size=12.,color="rgba(0, 0, 0, 1)", symbol="diamond-open")))
    i+=1

if dimension == "2D":
    mean_rslt, std_rslt = Measurement_devices.IFT8_inversion()
    print(mean_rslt)
    print(std_rslt)
    contour = [go.Contour(z=normalized_ground_truth_field.T, colorscale='inferno', contours_coloring='heatmap'),
               go.Contour(z=mean_rslt, colorscale='inferno', contours_coloring='heatmap'),
               go.Contour(z=np.abs(normalized_ground_truth_field.T-np.array(mean_rslt)), colorscale='inferno', contours_coloring='heatmap'),
               go.Contour(z=std_rslt, colorscale='inferno', contours_coloring='heatmap'),
               ]


    contour_plot = go.Figure(data=contour)
    Measuring_situation_plot = go.Figure(data=plot_data)

    plot_figure = make_subplots(rows=2, cols=3, specs=[[{"type": "scene", "rowspan": 2}, {}, {}], [{}, {}, {}]],
                                column_widths=[0.5, 0.25, 0.25],
                                subplot_titles=("3D Measuring Situation", "Ground Truth", "Retrieved Field", "",
                                                "Absolute Residual GrTr-RetrField", "Standard Deviation Retr. Field."))
    for t in Measuring_situation_plot.data:
        plot_figure.add_trace(t, row=1, col=1)


    plot_figure.add_trace(contour_plot.data[0], row=1, col=2)
    plot_figure.add_trace(contour_plot.data[1], row=1, col=3)
    plot_figure.add_trace(contour_plot.data[2], row=2, col=2)
    plot_figure.add_trace(contour_plot.data[3], row=2, col=3)

    plot_figure.update_layout(scene_xaxis_showticklabels=True,
                                scene_yaxis_showticklabels=True,
                                scene_zaxis_showticklabels=True,
                                showlegend=False)

    plot_figure.show()
else:
    mean_rslt, std_rslt = Measurement_devices.IFT8_inversion_3D()
    print(mean_rslt)
    print(std_rslt)
    results =  [go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                          value=vol.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                          value=mean_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                          value=vol.flatten()-mean_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                          value=std_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25)]

    results_plot = go.Figure(data=results)
    Measuring_situation_plot = go.Figure(data=plot_data)

    plot_figure = make_subplots(rows=2, cols=3, specs=[[{"type": "scene", "rowspan": 2}, {"type": "scene"}, {"type": "scene"}],
                                                       [{}, {"type": "scene"}, {"type": "scene"}]],
                                column_widths=[0.5, 0.25, 0.25],
                                subplot_titles=("3D Measuring Situation", "Ground Truth", "Retrieved Field", "",
                                                "Absolute Residual GrTr-RetrField", "Standard Deviation Retr. Field."))
    for t in Measuring_situation_plot.data:
        plot_figure.add_trace(t, row=1, col=1)

    plot_figure.add_trace(results_plot.data[0], row=1, col=2)
    plot_figure.add_trace(results_plot.data[1], row=1, col=3)
    plot_figure.add_trace(results_plot.data[2], row=2, col=2)
    plot_figure.add_trace(results_plot.data[3], row=2, col=3)

    plot_figure.update_layout(scene_xaxis_showticklabels=True,
                              scene_yaxis_showticklabels=True,
                              scene_zaxis_showticklabels=True,
                              showlegend=False)

    plot_figure.show()


















































