# Imports here to enable access to these functions from the library import without having to import those files.
# Effectively using this init in the same manner as a C header file. If there's a more pythonic way to do this,
# it should change to that.
import numpy as np

from pithermalcam import web_server
from pithermalcam.pi_therm_cam import PiThermalCam


def test_camera():
    """Check for an average temperature value to ensure the camera is connected and working."""
    # pylint: disable=protected-access
    try:
        thermcam = PiThermalCam()  # Instantiate class
        temp_c, temp_f = thermcam.get_mean_temp()
        print(thermcam._raw_image)
        print(type(thermcam._raw_image))
        print(thermcam._raw_image.shape)

        unique, counts = np.unique(thermcam._raw_image, return_counts=True)
        value_counts = dict(zip(unique, counts))
        zero_locs = np.argwhere(thermcam._raw_image == 0)
        print(f"zero loc: {zero_locs}")
        zero_vals = [thermcam._raw_image[x[0]][x[1]] for x in zero_locs]
        print(f"zeros: {zero_vals}")
        shape = thermcam._raw_image.shape
        surroundings = [
            thermcam._raw_image[
                max(0, x[0] - 1) : min(shape[0], x[0] + 2),
                max(0, x[1] - 1) : min(shape[1], x[1] + 2),
            ]
            for x in zero_locs
        ]
        print(f"surroundings: {surroundings}")
        print(f"surrounding size: {[s.size for s in surroundings]}")
        print(f"sums: {[np.sum(x) for x in surroundings]}")
        # for s in zero_locs:
        #     thermcam._raw_image[s[0], s[1]] = 999
        # print(thermcam._raw_image)
        print(value_counts)
        print("Camera seems to be connected and returning a value:")
        print("Average MLX90640 Temperature: {0:2.1f}C ({1:2.1f}F)".format(temp_c, temp_f))
        print("To verify it's working, change the average temperature")
        print(
            "(e.g. by hold your hand over the camera) and run again to verify that the average temperature has changed."
        )
    except ValueError as e:
        if str(e) == "No I2C device at address: 0x33":
            print("ERROR: Camera not found. There seems to be no device connected to I2C address 0x33.")
    except Exception as e:
        print("Camera didn't seem to work properly. Returned the following error:")
        raise e


def display_camera_live(output_folder: str = "/home/ed/cam_saved_snapshots/"):
    """Display the camera live onscreen"""
    thermcam = PiThermalCam(output_folder=output_folder)  # Instantiate class
    thermcam.display_camera_onscreen()


def stream_camera_online(output_folder: str = "/home/ed/cam_saved_snapshots/"):
    """Start a flask server streaming the camera live"""
    # This is a clunky way to do this, the better approach would likely to be restructuring web_server.py with the
    # Flask Blueprint approach
    # If the code were restructure for this, the code would be much more complex and opaque for running directly though
    web_server.start_server(output_folder=output_folder)


# Add attributes to existing pithermalcam object
setattr(PiThermalCam, "stream_camera_online", stream_camera_online)
