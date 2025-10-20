def fun1(x,y):
    """
    adds two input numbers, x and y.
    """
    return x+y

def fun2(x, y):
    """
    subtracts y from x.
    """
    return y-x

def fun3(x,y):
    """
    multiplies x and y.
    """
    return x*y

def fun4(x, y):
    """
    ombines the results of the above functions and returns their sum.
    """
    ans = fun1(x,y) + fun2(x,y) + fun3(x,y)
    return ans
