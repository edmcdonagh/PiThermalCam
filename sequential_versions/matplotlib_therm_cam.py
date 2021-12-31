#!/usr/bin/python3

##################################
# MLX90640 Thermal Camera w Raspberry Pi
##################################

import configparser
import logging
import time

import adafruit_mlx90640
import board
import busio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

profiling = False  # Flag to turn profiling on
if profiling:
    import cProfile
    import pstats

# Manual Params
DEBUG_MODE = False

# Set up Logger
if DEBUG_MODE:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(name)s:%(lineno)d] %(message)s", level=logging.DEBUG
    )
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # Disable warnings from matplotlib font manager when in debug mode
else:
    logging.basicConfig(
        filename="pithermcam.log",
        filemode="a",
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(name)s:%(lineno)d] %(message)s",
        level=logging.WARNING,
        datefmt="%d-%b-%y %H:%M:%S",
    )
logger = logging.getLogger(__name__)

# Parse Config file
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read("sequential_config.ini")
logger.debug(f"Config file sections found: {config.sections()}")

# Read Global variables from config file
# Note: Raw parsing used to avoid having to escape % characters in time strings
logger.debug("Reading config file and initializing variables...")
output_folder = config.get(section="FILEPATHS", option="output_folder", raw=True)

# Setup camera
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)  # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c)  # begin MLX90640 with I2C comm


def c_to_f(temp: float):
    """ Convert temperature from C to F """
    return (9.0 / 5.0) * temp + 32.0


# print out the average temperature from the MLX90640
def print_mean_temp():
    """
    Get mean temp of entire field of view. Return both temp C and temp F.
    """
    frame = np.zeros((24 * 32,))  # setup array for storing all 768 temperatures
    while True:
        try:
            mlx.getFrame(frame)  # read MLX temperatures into frame var
            break
        except ValueError:
            continue  # if error, just read again

    temp_c = np.mean(frame)
    temp_f = c_to_f(temp_c)
    print("Average MLX90640 Temperature: {0:2.1f}C ({1:2.1f}F)".format(temp_c, temp_f))
    return temp_c, temp_f


def simple_pic():
    """# -- 2Hz Sampling with Simple Routine"""
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ  # set refresh rate
    mlx_shape = (24, 32)

    # setup the figure for plotting
    # plt.ion() # enables interactive plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    therm1 = ax.imshow(np.zeros(mlx_shape), vmin=0, vmax=60)  # start plot with zeros
    cbar = fig.colorbar(therm1)  # setup colorbar for temps
    cbar.set_label(r"Temperature [$^{\circ}$C]", fontsize=14)  # colorbar label

    frame = np.zeros((24 * 32,))  # setup array for storing all 768 temperatures
    mlx.getFrame(frame)  # read MLX temperatures into frame var
    data_array = np.reshape(frame, mlx_shape)  # reshape to 24x32
    therm1.set_data(np.fliplr(data_array))  # flip left to right
    therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))  # set bounds
    cbar.on_mappable_changed(therm1)  # update colorbar range
    plt.pause(0.001)  # required
    fig.savefig(output_folder + "simple_pic.png", dpi=300, facecolor="#FCFCFC", bbox_inches="tight")


def simple_camera_read():
    """# -- Sampling with Simple Routine"""
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ  # set refresh rate
    mlx_shape = (24, 32)

    # setup the figure for plotting
    plt.ion()  # enables interactive plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    therm1 = ax.imshow(np.zeros(mlx_shape), vmin=0, vmax=60)  # start plot with zeros
    cbar = fig.colorbar(therm1)  # setup colorbar for temps
    cbar.set_label(r"Temperature [$^{\circ}$C]", fontsize=14)  # colorbar label

    frame = np.zeros((24 * 32,))  # setup array for storing all 768 temperatures
    t_array = []
    while True:
        t1 = time.monotonic()
        try:
            mlx.getFrame(frame)  # read MLX temperatures into frame var
            data_array = np.reshape(frame, mlx_shape)  # reshape to 24x32
            therm1.set_data(np.fliplr(data_array))  # flip left to right
            therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))  # set bounds
            cbar.on_mappable_changed(therm1)  # update colorbar range
            plt.pause(0.001)  # required
            # fig.savefig(output_folder + 'mlx90640_test_fliplr.png',dpi=300,facecolor='#FCFCFC', bbox_inches='tight') # comment out to speed up
            t_array.append(time.monotonic() - t1)
            print("Sample Rate: {0:2.1f}fps".format(len(t_array) / np.sum(t_array)))
        except ValueError:
            continue  # if error, just read again


def interpolated_pic():
    """# -- 2fps with Interpolation and Blitting"""
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ  # set refresh rate

    mlx_shape = (24, 32)

    mlx_interp_val = 10  # interpolate # on each dimension
    mlx_interp_shape = (mlx_shape[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val)  # new shape

    logger.debug("Setting up plot")
    fig = plt.figure(figsize=(12, 9))  # start figure
    ax = fig.add_subplot(111)  # add subplot
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of unnecessary padding
    therm1 = ax.imshow(
        np.zeros(mlx_interp_shape), interpolation="none", cmap=plt.cm.bwr, vmin=25, vmax=45
    )  # preemptive image
    cbar = fig.colorbar(therm1)  # setup colorbar
    cbar.set_label(r"Temperature [$^{\circ}$C]", fontsize=14)  # colorbar label

    fig.canvas.draw()  # draw figure to copy background
    ax_background = fig.canvas.copy_from_bbox(ax.bbox)  # copy background
    fig.show()  # show the figure before blitting

    frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 768 pts

    def plot_update():
        logger.debug("Updating plot")
        fig.canvas.restore_region(ax_background)  # restore background
        mlx.getFrame(frame)  # read mlx90640
        data_array = np.fliplr(np.reshape(frame, mlx_shape))  # reshape, flip data
        data_array = ndimage.zoom(data_array, mlx_interp_val)  # interpolate
        therm1.set_array(data_array)  # set data
        therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))  # set bounds
        cbar.on_mappable_changed(therm1)  # update colorbar range

        ax.draw_artist(therm1)  # draw new thermal image
        fig.canvas.blit(ax.bbox)  # draw background
        fig.canvas.flush_events()  # show the new image
        return

    plot_update()
    fig.savefig(output_folder + "interp_pic.png", dpi=300, facecolor="#FCFCFC", bbox_inches="tight")


def interpolated_camera_read():
    """# -- Higher fps with Interpolation and Blitting"""
    colorbar_update_interval = 5  # Seconds
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ  # set refresh rate

    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    mlx_shape = (24, 32)

    mlx_interp_val = 10  # interpolate # on each dimension
    mlx_interp_shape = (mlx_shape[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val)  # new shape

    fig = plt.figure(figsize=(9, 5))  # start figure
    ax = fig.add_subplot(111)  # add subplot
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of unnecessary padding
    therm1 = ax.imshow(
        np.zeros(mlx_interp_shape), interpolation="none", cmap=plt.cm.bwr, vmin=25, vmax=45
    )  # preemptive image
    cbar = fig.colorbar(therm1)  # setup colorbar
    cbar.set_label(r"Temperature [$^{\circ}$C]", fontsize=14)  # colorbar label

    fig.canvas.draw()  # draw figure to copy background
    ax_background = fig.canvas.copy_from_bbox(ax.bbox)  # copy background
    fig.show()  # show the figure before blitting

    frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 768 pts

    def plot_update(full_update: bool):
        fig.canvas.restore_region(ax_background)  # restore background
        mlx.getFrame(frame)  # read mlx90640
        data_array = np.fliplr(np.reshape(frame, mlx_shape))  # reshape, flip data
        data_array = ndimage.zoom(data_array, mlx_interp_val)  # interpolate
        if full_update:
            therm1.set_array(data_array)  # set data
            therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))  # set bounds
            cbar.on_mappable_changed(therm1)  # update colorbar range

        ax.draw_artist(therm1)  # draw new thermal image
        fig.canvas.blit(ax.bbox)  # draw background
        fig.canvas.flush_events()  # show the new image
        return

    t_array = []
    last_update = -100
    count = 0
    while True:
        t1 = time.monotonic()  # for determining frame rate
        # try:
        if t1 - last_update > colorbar_update_interval:
            last_update = t1
            full_update = True
        else:
            full_update = False
        plot_update(full_update)  # update plot
        # except:  # pylint: disable=E722
        #     continue
        # approximating frame rate
        t_array.append(time.monotonic() - t1)
        if len(t_array) > 10:
            t_array = t_array[1:]  # recent times for frame rate approx
        if full_update:
            print("Frame Rate: {0:2.1f}fps".format(len(t_array) / np.sum(t_array)))
            count += 1
            if profiling & (count >= 20):
                logger.info("Printing and dumping profiling stats")
                pr.disable()
                pr.dump_stats(output_folder + "profiling_stats.prof")
                ps = pstats.Stats(output_folder + "profiling_stats.prof")
                ps.sort_stats(pstats.SortKey.CUMULATIVE)
                ps.print_stats(50)
                break


if __name__ == "__main__":
    # Pick the mode to run in. Mode corresponds to functions in elif below.
    mode = 1

    if mode == 1:
        print_mean_temp()
    elif mode == 2:
        simple_pic()
    elif mode == 3:
        simple_camera_read()
    elif mode == 4:
        interpolated_pic()
    elif mode == 5:
        interpolated_camera_read()
    else:
        print("Incorrect or missing mode.")
