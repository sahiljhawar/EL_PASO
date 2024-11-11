from datetime import datetime

from examples.products.VanAllenProbes import RBSPICE_tofxeh, HOPE_electron
from examples.products.Arase import AraseMepi

hope = HOPE_electron('rbspa', './', './', 'IRBEM/libirbem.so')
#hope.download(datetime(2017,9,1), datetime(2017,9,1,23))
hope.process(datetime(2017,9,1), datetime(2017,9,1,23))

#rbspice = RBSPICE_tofxeh('rbspa', './', './', 'IRBEM/libirbem.so')

# #rbspice.download(datetime(2013, 3, 17), datetime(2013, 3, 18))
#rbspice.process(datetime(2013, 3, 17), datetime(2013, 3, 17, 23))

#arase = AraseMepi('./', './', 'IRBEM/libirbem.so')

# arase.download(datetime(2017, 4, 17), datetime(2017, 4, 18))
#arase.process(datetime(2017, 4, 17), datetime(2017, 4, 17, 23))
