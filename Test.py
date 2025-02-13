
from mpmath import mp
mp.dps = 100  # set number of digits

for i in range(1, 100):
    print("1/" + str(i) + ": "+ str(mp.mpf(1)/i))