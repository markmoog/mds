import numpy as np
import copy
from math import cos, sin, acos, tan, sqrt

def gradient_descent(positions, distances, rate):
    new_positions = copy.deepcopy(positions)
    error = 0

    for id_0, id_1, target_distance in distances:
        position_delta = positions[id_0] - positions[id_1]
        current_distance = np.sqrt(position_delta.dot(position_delta))
        gradient = (target_distance - current_distance) * position_delta

        new_positions[id_0] += rate * gradient
        new_positions[id_1] -= rate * gradient

        error += (target_distance - current_distance)**2

    return new_positions, error


def mds(distances, rate, iterations, positions=None, dimensions=2):
    if positions is None:
        # Randomly create initial positions. Use the average distance between
        # elements to determine the initial distribution of positions
        ids = set()
        total_distance = 0
        for id_0, id_1, distance in distances:
            ids.add(id_0)
            ids.add(id_1)
            total_distance += distance

        spread = total_distance / len(distances)
        positions = spread * np.random.randn(len(ids), dimensions)

    for i in range(iterations):
        positions, error = gradient_descent(positions, distances, rate)
        print(error)

    return positions


# Gradient descent algorithm for mds on an n-dimensional spherical surface
def gradient_descent_sphere(radius, positions, distances, rate):
    new_positions = copy.deepcopy(positions)
    error = 0

    for id_0, id_1, target_distance in distances:
        base_0 = 1
        base_1 = 1
        gamma = 0
        gradient_0 = np.zeros(positions[0].shape, dtype='f')
        gradient_1 = np.zeros(positions[0].shape, dtype='f')
        i = 0

        for coord_0, coord_1 in zip(positions[id_0], positions[id_1]):
            gamma_comp = base_0 * cos(coord_0) * base_1 * cos(coord_1)
            gamma += gamma_comp

            for dim in range(i):
                gradient_0[dim] += gamma_comp / tan(positions[id_0][dim])
                gradient_1[dim] += gamma_comp / tan(positions[id_1][dim])

            gradient_0[i] = -gamma_comp * tan(coord_0)
            gradient_1[i] = -gamma_comp * tan(coord_1)

            base_0 = base_0 * sin(coord_0)
            base_1 = base_1 * sin(coord_1)

            i += 1

        gamma += base_0 * base_1

        for dim in range(i):
            gradient_0[dim] += base_0 * base_1 / tan(positions[id_0][dim])
            gradient_1[dim] += base_0 * base_1 / tan(positions[id_1][dim])

        distance = abs(radius * acos(gamma))
        prefactor = (target_distance - distance) # / sqrt(1 - gamma**2)

        error += (target_distance - distance)**2

        new_positions[id_0] -= rate * prefactor * gradient_0
        new_positions[id_1] -= rate * prefactor * gradient_1

    return new_positions, error


# MDS on the surface of a sphere instead of n-dimensional euclidean space
def mds_sphere(radius, distances, rate, iterations, positions=None, dimensions=2):
    if positions is None:
        # Randomly create initial positions. Use the average distance between
        # elements to determine the initial distribution of positions
        ids = set()
        for id_0, id_1, distance in distances:
            ids.add(id_0)
            ids.add(id_1)

        # This doesn't create the best initial distribution on a sphereical
        # surface bu it will do.
        positions = 3.14159 * np.random.rand(len(ids), dimensions)

    print(positions)
    for i in range(iterations):
        positions, error = gradient_descent_sphere(radius, positions, distances, rate)
        print(error)

    return positions








