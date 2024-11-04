from datetime import datetime

from examples.products.VanAllenProbes import RBSPICE_tofxeh

rbspice = RBSPICE_tofxeh('rbspa', './', './')

#rbspice.download(datetime(2013, 3, 17), datetime(2013, 3, 18))
rbspice.process(datetime(2013, 3, 17), datetime(2013, 3, 17, 23))
