import socket
import json
import time
from typing import List


HOST = 'localhost'  # آدرس سرور یونیتی
PORT = 65432

# Define the Position class
class Position:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

# Define the Rotation class
class Rotation:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

# Define the Element class
class Element:
    def __init__(self, element_id: str, position: Position, rotation: Rotation):
        self.ElementId = element_id
        self.Position = position
        self.Rotation = rotation

    def to_dict(self):
        return {
            "ElementId": self.ElementId,
            "Position": self.Position.to_dict(),
            "Rotation": self.Rotation.to_dict()
        }

# Define the WrappedElements class
class Elements:
    def __init__(self, elements: List[Element]):
        self.Elements = elements

    def to_dict(self):
        return {"Elements": [element.to_dict() for element in self.Elements]}



# class Vector3D:
#     def __init__(self,x,y,z):
#         self.x=x
#         self.y=y
#         self.z=z

#     def toJSON(self):
#         return json.dumps(
#             self,
#             default=lambda o: o.__dict__, 
#             sort_keys=True,
#             indent=4).encode('utf-8')


# class element:
#   def __init__(self,elementId,position: Vector3D,rotation:Vector3D):
#       self.ElementId=elementId
#       self.Position=position
#       self.Rotation=rotation

#   def toJSON(self):
#         return json.dumps(
#             self,
#             default=lambda o: o.__dict__, 
#             sort_keys=True,
#             indent=4).encode('utf-8')


# class elements:
#   def __init__(self, elements:list[element]):
#       self.Elements=elements

#   def toJSON(self):
#         return json.dumps(
#             self,
#             default=lambda o: o.__dict__, 
#             sort_keys=True,
#             indent=4).encode('utf-8')


# elements = [
#     {"ElementId": "1",  # Example element ID
#     "Position": {"x": 1.0, "y": 2.0, "z": 3.0},
#     "Rotation": {"x": 0.0, "y": 90.0, "z": 0.0}}
#     ,
#     {"ElementId": "2",  # Example element ID
#     "Position": {"x": 1.0, "y": 1.0, "z": 1.0},
#     "Rotation": {"x": 20.0, "y": 0.0, "z": 0.0}}

# ]

# Wrap the list in a dictionary under the key "elements"
# WrapedElements = {"Elements": elements}

# def send_positions():
#     while True:
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 s.connect((HOST, PORT))
#                 print("Connected to server")
#                 while True:
#                     data = json.dumps(WrapedElements).encode('utf-8')
#                     s.sendall(data)
#                     time.sleep(1)  # ارسال داده‌ها هر ثانیه
#         except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
#             print(f"Connection failed: {e}")
#             time.sleep(5)  # تلاش مجدد بعد از 5 ثانیه
#         finally:
#             s.close()  # بستن اتصال

def connect_to_server():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        print("Connected to server")
        return s
    except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
        print(f"Connection failed: {e}")
        return None

# def send_positions_single(_elements: Elements):
#         print("aaa")

#     #while True:
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 s.connect((HOST, PORT))
#                 print("Connected to server")
#                 #while True:
#                     #data = json.dumps(_elements).encode('utf-8')
#                     #data=bytes(_elements.toJSON(),'utf-8')
#                     #print(_elements.toJSON())
#                     #s.sendall(data)
#                 print(json.dumps(_elements.to_dict(), indent=4))
#                 s.sendall(json.dumps(_elements.to_dict(), indent=4).encode())
#                 #time.sleep(1)  # ارسال داده‌ها هر ثانیه
#         except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
#             print(f"Connection failed: {e}")
#             time.sleep(5)  # تلاش مجدد بعد از 5 ثانیه
#         finally:
#             s.close()  # بستن اتصال

# # if __name__ == "__main__":
# #     send_positions()


def send_positions_single(sock, _elements: Elements):
    try:
        sss=json.dumps(_elements.to_dict(), indent=4)
        zzz=json.dumps(_elements.to_dict(), indent=4).encode()
        print(json.dumps(_elements.to_dict(), indent=4))
        sock.sendall(zzz)
    except (ConnectionResetError, BrokenPipeError) as e:
        print(f"Connection failed during send: {e}")
        sock.close()
        return False
    return True