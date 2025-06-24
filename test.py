from dynamixel_sdk import *

portHandler = PortHandler('COM5')
packetHandler = PacketHandler(2.0)

if portHandler.openPort():
    print("Port opened")
    if portHandler.setBaudRate(1000000):
        print("Baudrate set")

        for id in range(1, 7):
            dxl_comm_result, dxl_error = packetHandler.ping(portHandler, id)
            if dxl_comm_result == 0:
                print(f"Motor {id} responded.")
            else:
                print(f"Motor {id} no response.")

        portHandler.closePort()
    else:
        print("Baudrate setting failed.")
else:
    print("Port open failed.")
    
