def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    """
    Perform matrix-vector multiplication.

    This function multiplies a matrix 'a' with a vector 'b'. It checks if the 
    dimensions are compatible for multiplication and returns the resulting vector.

    Parameters:
    a (list[list[int|float]]): The matrix, represented as a list of lists.
                               Each inner list is a row of the matrix.
    b (list[int|float]): The vector to be multiplied with the matrix.

    Returns:
    list[int|float]: The resulting vector after multiplication.
                     Returns -1 if the dimensions are incompatible.

    Notes:
    - The function checks if the number of columns in 'a' matches the length of 'b'.
    - If dimensions are incompatible, it returns -1 instead of raising an exception.
    - The result is a vector with the same number of elements as rows in 'a'.

    Example:
    >>> matrix = [[1, 2], [3, 4]]
    >>> vector = [5, 6]
    >>> result = matrix_dot_vector(matrix, vector)
    >>> print(result)
    [17, 39]
    """
    if len(a[0]) != len(b):
        return -1
    
    vals = []
    for i in a:
        hold = 0
        for j in range(len(i)):
            hold += (i[j] * b[j])
        vals.append(hold)
    
    return vals