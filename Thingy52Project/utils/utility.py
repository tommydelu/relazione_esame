from bleak import *
from utils.UUIDs import  *
import struct
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


async def scan():
    """
    Scan for BLE devices
    :return: A list of BLE devices foun
    """
    devices = await BleakScanner.discover(cb=dict(use_bdaddr=True))

    return devices

def find(discovered_devices, addresses):

    my_devices = []
    for i in range(len(discovered_devices)):
        if discovered_devices[i].address in addresses:
            my_devices.append(discovered_devices[i])
            # Avoid useless checks
            if len(my_devices) == len(addresses):
                break

    return my_devices

async def change_status(client, status: str):
    """
    Change the color of the LED on the Thingy52
    :param client: The client object
    :param status: The status to change to
    :return: None
    """
    payload = None

    if status == "connected":
        format_str = "<4B"
        constant_light = 1
        green = (0, 255, 0)
        payload = struct.pack(format_str, constant_light, *green)

    if status == "recording":
        format_str = "<4B"
        constant_light = 1
        red = (255, 0, 0)
        payload = struct.pack(format_str, constant_light, *red)

    if payload is not None:
        # Write the color data to the LED characteristic
        await client.write_gatt_char(UIS_LED_UUID, payload)

def motion_characteristics(
        step_counter_interval=100,
        temperature_comp_interval=100,
        magnetometer_comp_interval=100,
        motion_processing_unit_freq=60,
        wake_on_motion=1
):
    """

    :param step_counter_interval:
    :param temperature_comp_interval:
    :param magnetometer_comp_interval:
    :param motion_processing_unit_freq:
    :param wake_on_motion:
    :return:
    """

    """
    The format string '<4H B' specifies the following:
        <  : Little-endian byte order.
        4H : Four 16-bit unsigned integers.
        B  : One 8-bit unsigned integer.
    """
    format_str = "<4H B"

    return struct.pack(format_str,
                       step_counter_interval,
                       temperature_comp_interval,
                       magnetometer_comp_interval,
                       motion_processing_unit_freq,
                       wake_on_motion)

def cm_analysis(y_true, y_pred, filename, labels, classes, ymap=None, fig_size=(17, 14), specific_title=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=2)

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            #elif c == 0:
            #    annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=fig_size)
    plt.yticks(va='center')

    sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format': PercentFormatter()}, yticklabels=classes, cmap="Blues")

    plot_title = filename.split('/')[-2]
    if specific_title is None:
        pass
    else:
        plt.title(specific_title, fontsize=40, fontweight="bold")

    plt.subplots_adjust(hspace=0.5, top=2.88)
    plt.tight_layout()

    plt.savefig(f"{filename}.png",  bbox_inches='tight', dpi=300)
    cm.to_csv(f"{filename}.csv")
    plt.close()

def get_uuid(device):
    """
    Get the MAC address of a device, useful in macOS
    :param device: BLE Device
    :return: MAC address
    """
    return str(device.details[0].identifier).split(",")[1].split("= ")[1]