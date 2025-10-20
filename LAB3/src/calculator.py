def fun1(x, y):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("Both x and y must be numbers")
    return x + y

def fun2(x, y):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("Both x and y must be numbers")
    return y - x

def fun3(x, y):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("Both x and y must be numbers")
    return x * y

def fun4(x, y):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("Both x and y must be numbers")
    return fun1(x, y) + fun2(x, y) + fun3(x, y)
