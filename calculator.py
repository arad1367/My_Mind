def multiply_p(*numbers):
    total = 1
    for i in numbers:
        total *= i
    print(total)


multiply(3, 5, 10)
multiply(3, 5, 20)
