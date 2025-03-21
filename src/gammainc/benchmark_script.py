from gammainc import custom_gammaincc

# This script calls the custom gammaincc function 1 million times.
# Gages how long these many calls take with a stable input.

sum = 0

for i in range(0, 1000000):
    s = -1.2
    x = 2
    value = custom_gammaincc(s, x)
    sum += value

print(sum)
