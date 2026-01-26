compute=[739, 1880, 13366, 19169]
trans=[2341, 1777, 5465, 4817]
energy=[797095, 779665, 521769, 552560]
video=[45173386370, 44357165608, 55063940552, 55133457070]

GAMMA1 = 1.0 / 10
GAMMA2 = 1.0 / 6
GAMMA3 = 1.0 / 800
GAMMA4 = -1.0 / 100000000

o=[]

for i in range(len(compute)):
    o.append(trans[i] * GAMMA1 + compute[i] * GAMMA2 + energy[i] * GAMMA3 + video[i] * GAMMA4)

print(1)