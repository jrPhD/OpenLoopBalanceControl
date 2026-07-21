# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:04:02 2026

@author: ronne
"""



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

from IPython.display import HTML

from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter, FFMpegWriter



from symbrim.utilities.plotting import Plotter


def animate_solution(t_simu, x_opt, r_opt, bicycle, x, r, p, ani_name):

    # Create some functions to interpolate the results.
    x_eval = CubicSpline(t_simu, x_opt)
    r_eval = CubicSpline(t_simu, r_opt)
    # dis_eval = CubicSpline(t_simu, dis.T)
    # max_disturbance = dis.max()

    # Plot the initial configuration of the model
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
    plotter = Plotter.from_model(bicycle, ax=ax)
    # plotter.add_vector(
    #     disturbance * bicycle.rear_frame.wheel_hub.axis / max_disturbance,
    #     bicycle.rear_frame.saddle.point,
    #     name="disturbance",
    #     color="r",
    # )
    # plotter.lambdify_system((x, r, disturbance, param))
    # plotter.evaluate_system(x_eval(0.0), r_eval(0.0),dis[0], param_vals)
    param, param_vals = zip(*p.items())
    plotter.lambdify_system((x, r, param))
    plotter.evaluate_system(x_eval(0.0), r_eval(0.0), param_vals)
    plotter.plot()
    X, Y = np.meshgrid(np.arange(-1, 10, 0.5), np.arange(-1, 3, 0.5))
    ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1, cstride=1)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.view_init(19, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    fps = 30
    # ani = plotter.animate(
    #     lambda ti: (x_eval(ti), r_eval(ti), dis_eval(ti), param_vals), frames=np.arange(0, t_simu[-1], 1 / fps), blit=False
    # )
    
    ani = plotter.animate(
        lambda ti: (x_eval(ti), r_eval(ti), param_vals), frames=np.arange(0, t_simu[-1], 1 / fps), blit=False
    )
    
    display(HTML(ani.to_jshtml(fps=fps)))
    
    html_writer = HTMLWriter()
    ani.save(ani_name if ani_name.endswith(".html") else ani_name + ".html", writer=html_writer)
