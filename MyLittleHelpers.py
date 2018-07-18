

def prod(x_list):
    """
    Calculates the product of all elements in an list.
    :param x: list or tupel of scalars
    :return: Product of all scalars
    """
    y = 1
    for x in x_list:
        y *= x
    return y

def sep():
    print('\n{}\n'.format('-'*70))