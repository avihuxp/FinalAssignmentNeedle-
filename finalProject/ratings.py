from typing import List, Dict
import numpy as np


def manhattan_distance(x: int, y: int) -> int:
    """
    Calculate the Manhattan distance between two integers.

    :param x: The first integer.
    :param y: The second integer.
    :return: The Manhattan distance between x and y.
    """
    return abs(x - y)



def calculate_community_statistics(communities: List[Dict[int, int]]) -> \
        List[Dict[int, float]]:
    """
    Calculate the standard deviation of Manhattan distances within
    each community.
    :param communities: A list of dictionaries where each dictionary
                        represents a community.
                        Each key in the dictionary is an integer (node id),
                        and the value is the ranking.
    :return: A list of dictionaries, where each dictionary contains the
             size of the community as the key and the standard deviation of
             the Manhattan distances within the community as the value.
    """
    result = []
    for community in communities:
        community_size = len(community)
        values = list(
            community.values())
        # Calculate the Manhattan distances between all pairs of values.
        distances = []
        for i in range(community_size):
            for j in range(i + 1, community_size):
                distance = manhattan_distance(values[i], values[j])
                distances.append(distance)
        # Calculate the standard deviation if there are distances
        # to compute.
        if distances:
            # Standard deviation of the Manhattan distances.
            std_dev = np.std(distances)
        else:
            # If there are no distances, set standard deviation to 0.
            std_dev = 0
        # Append a dictionary with community size as key and standard
        # deviation as value
        result.append({community_size: std_dev})
    return result

def main():
    communities = [
    {1: 10, 2: 15, 3: 20},  # Community 1
    {4: 5, 5: 8, 6: 12, 7: 14}]
    output = calculate_community_statistics(communities)
    print(output)


if __name__ == '__main__':
    main()
