import random

import torch

import pickle
import os

from config import speed

coords_file_path = file_path = os.path.join(os.getcwd(), "..", "coords data")
pickle_in = open(coords_file_path, 'rb')
coords_data = pickle.load(pickle_in)

def get_random_problems(batch_size, problem_size):
    depot_x = 5.622153766066174
    depot_y = 52.0308709742657
    depot_xy = torch.tensor([depot_x,depot_y]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)
    depot_time_window = torch.tensor([0, 1]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)

    node_xy =torch.zeros((batch_size,problem_size,2))
    for x in range(batch_size):
        node_xy[x] = torch.tensor(coords_data.sample(n=problem_size,
                                                     replace=False).values)
    # shape: (batch, problem, 2)

    demand_scaler = 25000

    rate = 1/382
    node_demand = torch.distributions.Exponential(rate).sample((batch_size, problem_size)) / demand_scaler
    # shape: (batch, problem)

    p = 0.25
    # Create tensor of probabilities
    samples = torch.rand((batch_size, problem_size, 1)) < p

    tw_scalar = 14
    horizon_start = 5
    lower_tw = torch.tensor([5],dtype=torch.float32).repeat(batch_size, problem_size, 1)
    upper_tw = torch.tensor([19],dtype=torch.float32).repeat(batch_size, problem_size, 1)
    lower_tw[samples] = (torch.randint(15, 21, (batch_size, problem_size, 1))/ 2)[samples]
    upper_tw[samples] = (torch.randint(30, 38, (batch_size, problem_size, 1))/ 2)[samples]
    time_windows = torch.cat((lower_tw, upper_tw), dim=2)
    time_windows = (time_windows - horizon_start) / tw_scalar

    empty_tensor = torch.empty(batch_size, problem_size)
    service_times = torch.nn.init.trunc_normal_(empty_tensor, mean=0.23, std=0.24,
                                                a=0.05, b=0.6)
    p=0.02
    # Create tensor of probabilities
    samples = torch.rand((batch_size, problem_size)) < p
    service_times[samples] = (torch.randint(100,151, (batch_size,problem_size))/100)[samples]
    service_times = service_times/tw_scalar

    travel_times = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    prices = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    duals = torch.zeros((batch_size, problem_size+1))
    for x in range(batch_size):
        coords = torch.cat((depot_xy[x], node_xy[x]), 0)
        #distance = torch.cdist(coords,coords,p=2)
        distances = manhattan_geo_distance_matrix(coords) #torch.cdist(coords, coords, p=2)
        '''speeds = 60*(distances/10)
        speeds[speeds<35] = 35
        speeds[speeds>80]=  80'''
        travel_times[x] = distances/speed 
        travel_times[x].fill_diagonal_(0)
        duals[x] = create_duals(travel_times[x])
        prices[x] = (travel_times[x] - duals[x]) * -1
        prices[x].fill_diagonal_(0)
        min_val = torch.min(prices[x]).item()
        max_val = torch.max(prices[x]).item()
        prices[x] = prices[x] / max(abs(max_val), abs(min_val))

    travel_times = travel_times / tw_scalar
    duals = duals[:,1:] / tw_scalar
    time_windows = repair_time_windows(travel_times, time_windows, service_times,
                                       tw_scalar, 4)

    depot_xy[:,:,0] = (depot_xy[:,:,0]-4)/3
    depot_xy[:, :, 1] = (depot_xy[:, :, 1] - 51) / 2.5
    node_xy[:, :, 0] = (node_xy[:, :, 0] - 4) / 3
    node_xy[:, :, 1] = (node_xy[:, :, 1] - 51) / 2.5


    return depot_xy, node_xy, node_demand, time_windows, depot_time_window, duals, service_times, travel_times, prices


def create_duals(time_matrix):
    problem_size = time_matrix.shape[1]-1
    duals = torch.zeros(size=(problem_size+1,), dtype=torch.float32)
    indices = list(range(1, problem_size+1))
    scaler = 0.2 + 0.9 * torch.rand([]) #0.75*torch.rand([])
    non_zeros = random.randint(int(problem_size / 2), problem_size)
    chosen = random.sample(indices, non_zeros)
    max_travel_times, _ = torch.max(time_matrix,dim=0)
    randoms = torch.rand(size=(non_zeros,))
    duals[chosen] = max_travel_times[chosen] * scaler * randoms
    return duals

def repair_time_windows(travel_times, time_windows, service_times, tw_scalar,
                        min_tw_width, factor=0.0001):

    scaled_tw_width = min_tw_width / tw_scalar
    problem_size = travel_times.shape[1] - 1
    batch_size = travel_times.shape[0]

    latest_possible_arrivals = torch.ones(batch_size, problem_size) - travel_times[:, 1:, 0] - service_times
    latest_possible_arrivals -= factor
    time_windows[:, :, 1] = torch.minimum(time_windows[:, :, 1], latest_possible_arrivals)
    mask = time_windows[:, :, 1] - time_windows[:, :, 0] < scaled_tw_width

    time_windows[:, :, 0][mask] = time_windows[:, :, 1][mask] - scaled_tw_width
    return time_windows

def manhattan_geo_distance_matrix(coords):
    xp = torch
    arr = coords.clone().detach().to(dtype=torch.float32)

    lon_deg = arr[:, 0]
    lat_deg = arr[:, 1]

    # Pairwise degree diffs
    lon1 = lon_deg[:, None]; lon2 = lon_deg[None, :]
    lat1 = lat_deg[:, None]; lat2 = lat_deg[None, :]

    dlon_deg = xp.abs(lon2 - lon1)
    dlat_deg = xp.abs(lat2 - lat1)

    # Average latitude (radians) for E–W scaling
    lat_mean_rad = ((lat1 + lat2) * (xp.pi / 180.0)) * 0.5

    # Convert degree diffs to meters (approx.)
    # 1 deg lat ≈ 111,132 m; 1 deg lon ≈ 111,320 * cos(lat) m
    dy = 111_132.0 * dlat_deg
    dx = 111_320.0 * xp.cos(lat_mean_rad) * dlon_deg

    D = dx + dy  # Manhattan distance
    D = D / 1000.0
    return D

def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

def main():
    coords =torch.tensor([[ 5.62215377, 52.03087097],
 [ 4.914584  , 51.8450816 ],
 [ 4.779223  , 51.8676869 ],
 [ 4.6954949 , 51.887344  ],
 [ 4.691139 ,  51.8925349 ]])

    travel_times =torch.zeros((5,5),dtype=torch.float32)
    distances = manhattan_geo_distance_matrix(coords)  # torch.cdist(coords, coords, p=2)
    travel_times[distances < 10] = distances[distances < 10] / 35
    travel_times[distances > 10] = distances[distances > 10] / 70
    print(distances)
    print(travel_times)


if __name__ == "__main__":
    main()