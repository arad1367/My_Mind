def multiply_p(*numbers):
    total = 1
    for i in numbers:
        total *= i
    print(total)


multiply(3, 5, 10)


def add_p (x, y):
    a = lambda x,y:x+y
    return a(x,y)
