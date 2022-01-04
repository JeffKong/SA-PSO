def fitness(x):
    F = 100 * (x[0]**2 - x[1])**2 + (1 - x[0])**2
    return F
