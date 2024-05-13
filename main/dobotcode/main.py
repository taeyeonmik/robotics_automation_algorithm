from serial.tools import list_ports
# import pydobot
from dobot import Dobot

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    available_ports = list_ports.comports()
    available_ports.sort()
    print(f'available ports: {[x.device for x in available_ports]}')
    # print(f'name: {[x.name for x in available_ports]}')
    # print(f'desc: {[x.description for x in available_ports]}')


    port = available_ports[1].device

    # instantiation de Dobot avec un port donné
    device = Dobot(port=port, verbose=True)
    (x, y, z, r, j1, j2, j3, j4) = device.pose()
    print(f'x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}')

    """
        manipulation de robot
    """
    # init pose
    device.move_to(x + 50, y, z, r, wait=False)
    device.move_to(x, y + 50, z, r, wait=False)
    device.move_to(x, y, z, r, wait=False)
    device.move_to(x, y, z + 50, r, wait=False)
    device.wait(1000)

    device.move_to(x, y, z, r, wait=True)  # we wait until this movement is done before continuing
    device.close()