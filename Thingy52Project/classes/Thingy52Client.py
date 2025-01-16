import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from bleak import BleakClient, BLEDevice
from datetime import datetime
import struct
import onnxruntime as ort
from utils.UUIDs import TMS_RAW_DATA_UUID, TMS_CONF_UUID
from utils.utility import motion_characteristics, change_status, get_uuid
import numpy as np


class Thingy52Client(BleakClient):

    def __init__(self, device: BLEDevice):
        super().__init__(get_uuid(device))
        self.mac_address = device.address
        self.choice = ["record","train","predict"]

        # Recording information
        self.recording_name = None
        self.file = None

        self.model = ort.InferenceSession('training/CNN_180.onnx')
        self.classes = ["balling", "shooting"]
        self.buffer_size = 180
        self.data_buffer = []





    async def connect(self, **kwargs) -> bool:
        """
        Connect to the Thingy52 device
        :return: True if the connection is successful, False otherwise
        """
        print(f"Connecting to {self.mac_address}")
        await super().connect(**kwargs)

        try:
            print(f"Connected to {self.mac_address}")
            # Perform operations with the client here
            await change_status(self, "connected")
            return True

        except Exception as e:
            print(f"Failed to connect to {self.mac_address}: {e}")
            return False

    def save_to(self, file_name):
        self.recording_name = f"{self.mac_address.replace(':', '-')}_{file_name}.csv"

    async def disconnect(self) -> bool:
        """
        Disconnect from the Thingy52 device
        :return: True if the disconnection is successful, False otherwise
        """
        print(f"\nDisconnecting from {self.mac_address}")
        self.file.close()
        return await super().disconnect()




    def raw_data_callback(self, sender, data):
        # Handle the incoming accelerometer data here
        receive_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Accelerometer
        acc_x = (struct.unpack('h', data[0:2])[0] * 1.0) / 2 ** 10
        acc_y = (struct.unpack('h', data[2:4])[0] * 1.0) / 2 ** 10
        acc_z = (struct.unpack('h', data[4:6])[0] * 1.0) / 2 ** 10

        # Gyroscope
        gyro_x = (struct.unpack('h', data[6:8])[0] * 1.0) / 2 ** 5
        gyro_y = (struct.unpack('h', data[8:10])[0] * 1.0) / 2 ** 5
        gyro_z = (struct.unpack('h', data[10:12])[0] * 1.0) / 2 ** 5

        # Save the data to a file
        #self.file.write(f"{receive_time},{acc_x},{acc_y},{acc_z},{gyro_x},{gyro_y},{gyro_z}\n")
        #print(f"\r{self.mac_address} | {receive_time} - Accelerometer: X={acc_x: 2.3f}, Y={acc_y: 2.3f}, Z={acc_z: 2.3f}", end="", flush=True)

        # Update the data buffer
        if len(self.data_buffer) == self.buffer_size:
            input_data = np.array(self.data_buffer, dtype=np.float32).reshape(1, self.buffer_size, 6)
            input_ = self.model.get_inputs()[0].name

            predictions = self.model.run(None, {input_: input_data})[0]

            cls_index = np.argmax(self.model.run(None, {input_: input_data})[0], axis=1)[0]
            cls_probability = predictions[0][cls_index]
            print(f"\r{self.mac_address} | {receive_time} - Prediction: {self.classes[cls_index]} ({cls_probability * 100:.2f}%)", end="", flush=True)
            self.data_buffer.clear()

        self.data_buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])


    async def receive_inertial_data(self, sampling_frequency: int = 60):
        """
        Receive data from the Thingy52 device
        :return: None
        """
        # Set the sampling frequency
        payload = motion_characteristics(motion_processing_unit_freq=sampling_frequency)
        await self.write_gatt_char(TMS_CONF_UUID, payload)
        # Open the file to save the data

        directory = "training/data"
        file_path = os.path.join(directory, self.recording_name)
        self.file = open(file_path, "a+")


        # Ask to activate the raw data (Inertial data)
        await self.start_notify(TMS_RAW_DATA_UUID, self.raw_data_callback)

        # Change the LED color to red, recording status
        await change_status(self, "recording")

        try:
            while True:
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            await self.stop_notify(TMS_RAW_DATA_UUID)
            await self.disconnect()
            print("Stopped notification")