"""
Variational Framework for 2D.


"""

import numpy as np
import signal
import sys
from types import FrameType
from flask import Flask,request,jsonify
from utils.logging import logger
from fdtt_2D.image import depixelation_with_boundary_correction

app = Flask(__name__)


@app.route("/")
def calculate_depixelation() -> np.ndarray:
    """
    Calculates Dexpixelation of a sub_image in string form
    with the number of rows and columns specified by
    m, n and channels.

    :return:
    """

    values = request.args.get('sub_image')

    m = request.args.get('m')
    n = request.args.get('n')
    channels = request.args.get('channels')

    print("Logging...")
    print("\n")
    print("Values: ", values)
    print("\n")
    print("M: ", m)
    print("N: ", n)

    m = int(m)
    n = int(n)
    channels = int(channels)
    values = list(map(lambda x: float(x), values.split(",")))
    flat_arr = np.array(values)


    print("Flat array: ", flat_arr)
    print("Flat array shape: ", flat_arr.shape)

    arr_3d = flat_arr.reshape((m, n, channels))

    print("3D array: ", arr_3d)

    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # https://cloud.google.com/run/docs/logging#correlate-logs
    logger.info("Child logger with trace Id.")

    result = depixelation_with_boundary_correction(arr_3d)
    return result


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
