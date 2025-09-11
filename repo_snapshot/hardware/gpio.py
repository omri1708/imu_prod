# imu_repo/hardware/gpio.py
from __future__ import annotations
from typing import Any

class ResourceRequired(Exception): ...

# --- GPIO ---
try:
    import RPi.GPIO as RGPIO
    HAS_GPIO=True
except ImportError:
    HAS_GPIO=False

class GPIO:
    BCM="BCM"; OUT="OUT"; IN="IN"
    def __init__(self):
        if not HAS_GPIO:
            raise ResourceRequired("RPi.GPIO library required for GPIO access")

    def setup(self,pin:int,direction:str):
        mode=RGPIO.BCM if direction==self.BCM else RGPIO.BOARD
        RGPIO.setmode(mode)
        RGPIO.setup(pin,RGPIO.OUT if direction==self.OUT else RGPIO.IN)

    def write(self,pin:int,val:int): RGPIO.output(pin,val)
    def read(self,pin:int)->int: return RGPIO.input(pin)

# --- I²C ---
try:
    import smbus2
    HAS_I2C=True
except ImportError:
    HAS_I2C=False

class I2C:
    def __init__(self,bus:int=1):
        if not HAS_I2C:
            raise ResourceRequired("smbus2 required for I2C access")
        self.bus=smbus2.SMBus(bus)
    def write_byte(self,addr:int,val:int): self.bus.write_byte(addr,val)
    def read_byte(self,addr:int)->int: return self.bus.read_byte(addr)

# --- SPI ---
try:
    import spidev
    HAS_SPI=True
except ImportError:
    HAS_SPI=False

class SPI:
    def __init__(self,bus:int=0,dev:int=0):
        if not HAS_SPI:
            raise ResourceRequired("spidev required for SPI access")
        self.dev=spidev.SpiDev(); self.dev.open(bus,dev)
    def xfer(self,data:list[int])->list[int]: return self.dev.xfer(data)
