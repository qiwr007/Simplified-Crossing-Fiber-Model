from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
a = 0
for i in range(1000):
    for j in range(2000):
        a += 1
here = datetime.now()

print("run time =", here - now)
