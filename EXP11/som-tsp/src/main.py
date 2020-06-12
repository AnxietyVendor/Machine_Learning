from sys import argv

import numpy as np

from io_helper import read_tsp, normalize
from neuron import generate_network, get_neighborhood, get_route
from distance import select_closest, euclidean_distance, route_distance
from plot import plot_network, plot_route

def main():
    # 从命令行读取参数
    # 检查调用命令是否正确
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        return -1
    # 读取tsp文件名(python不作为参数关键字)
    problem = read_tsp(argv[1])
    # 训练SOM网络找到路径
    route = som(problem, 100000)

    problem = problem.reindex(route)

    distance = route_distance(problem)

    print('Route found of length {}'.format(distance))


def som(problem, iterations, learning_rate=0.8):
    '''
    Solve the TSP using a Self-Organizing Map.
    :params problem(dataframe): 城市坐标 
    :params iterations(int): 最大迭代次数
    :learning_rate(float): 学习率
    :return route
    '''    
    cities = problem.copy()
    
    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    # <Hyperparameter:times>
    n = cities.shape[0] * 8

    # Generate an adequate network of neurons:
    # (n,2)
    network = generate_network(n)
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):
        if not i % 100:
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")
        # Choose a random city
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        # <Hyperparameter:radius>
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        # Update the network's weights (closer to the city)
        network += gaussian[:,np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        # <Hyperparameter:decay rate>
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        # Check for plotting interval
        if not i % 1000:
            plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network(cities, network, name='diagrams/final.png')

    route = get_route(cities, network)
    plot_route(cities, route, 'diagrams/route.png')
    return route

if __name__ == '__main__':
    main()

# case: max_iter = 100000, lr = 0.8, radix = n//10
# n 1 2 3 4 5 6 7 8 9 10
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    