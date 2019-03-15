import time


def fun():
    print("hello world")
    time.sleep(1.3)


for t in range(3):
    fun()

for t in range(9):
    print(t*t)
    if t > 4:
        print("we in here")
    else:
        print('we not in here')

x = True

if x:
    print("works")
