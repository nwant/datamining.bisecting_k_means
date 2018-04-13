import math
import random


def euclidian_distance(p):
    return math.sqrt(sum([(x - y)**2 for x, y in p]))


def manhattan_distance(p):
    return sum([abs(x - y) for x, y in p])


def gen_random_points():
    return [(random.uniform(1, 100), random.uniform(1, 100)) for _ in range(20)]


def bisecting_k_means(k, p=gen_random_points()):
    c = [set(p)] # list of clusters. initial cluster`


