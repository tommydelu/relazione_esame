from classes.Thingy52Client import Thingy52Client
from utils.utility import scan, find
import asyncio


async def main():

    my_thingy_addresses = ["F1:58:6C:E2:D8:44"]
    discovered_devices = await scan()
    my_devices = find(discovered_devices, my_thingy_addresses)
    thingy52 = Thingy52Client(my_devices[0])
    await thingy52.connect()


    thingy52.save_to(str(input("Enter recording name: ")))
    await thingy52.receive_inertial_data()




if __name__ == '__main__':
    asyncio.run(main())