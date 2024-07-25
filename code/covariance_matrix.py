def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """
    Calculate the covariance matrix for a set of feature vectors.

    This function computes the covariance matrix for a given set of feature vectors. 
    The covariance matrix is a square matrix where each element (i,j) represents 
    the covariance between the i-th and j-th features.

    Parameters:
    vectors (list[list[float]]): A list of feature vectors. Each inner list represents
                                 a feature, and its elements are the observations for that feature.

    Returns:
    list[list[float]]: The computed covariance matrix.

    Notes:
    - The function assumes that all feature vectors have the same number of observations.
    - The covariance matrix is symmetric, so only the upper triangular part is computed
      and then mirrored to the lower triangular part.
    - The calculation uses the formula: cov(X,Y) = E[(X - E[X])(Y - E[Y])]
    - The result is divided by (n_observations - 1) for an unbiased estimate.

    Example:
    >>> vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    >>> cov_matrix = calculate_covariance_matrix(vectors)
    >>> print(cov_matrix)
    [[1.0, 1.0], [1.0, 1.0]]
    """
    n_features = len(vectors)
    n_observations = len(vectors[0])
    
    # Initialize the covariance matrix with zeros
    covariance_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]
    
    # Calculate means for each feature
    means = [sum(feature) / n_observations for feature in vectors]
    
    # Calculate covariances
    for i in range(n_features):
        for j in range(i, n_features):
            # Compute covariance between feature i and feature j
            covariance = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) 
                             for k in range(n_observations)) / (n_observations - 1)
            
            # Set covariance in both (i,j) and (j,i) positions due to symmetry
            covariance_matrix[i][j] = covariance_matrix[j][i] = covariance
    
    return covariance_matrix