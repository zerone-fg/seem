import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def getHighDistanceMatrix(datas):
    N, D = datas.shape
    dists = torch.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         vi = datas[i, :]
    #         vj = datas[j, :]
    #         dists[i, j] = torch.norm(vi - vj, 2)
    xx = torch.pow(datas, 2).sum(1, keepdim=True).expand(N, N)
    yy = torch.pow(datas, 2).sum(1, keepdim=True).expand(N, N).t()
    dist = xx + yy
    dist.addmm_(1, -2, datas, datas.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def get_density(dists, lambda1=50):
    N = dists.shape[0]
    s_dis = torch.zeros((N * N))
    index = torch.zeros((N, 2))  #### 4096, 2
    roi_dens = torch.zeros(N)
    for i in range(N):
        index[i] = torch.tensor((math.floor(i / 64), i - math.floor(i / 64) * 64))
    
    index_1 = index.unsqueeze(0).repeat(N, 1, 1)
    index_2 = index.unsqueeze(1).repeat(1, N, 1)
    index_3 = index_1 - index_2

    index_3 = index_3.view(-1, 2)

    for i in range(N * N):
        s_dis[i] = abs(index_3[i][0]) + abs(index_3[i][1])

    for i in range(N):
        for j in range(N):
            s_dis[i, j] = (s_dis[i, j] / lambda1 * 0.1 + 0.9) if s_dis[i, j] <= lambda1 else np.inf

    for i in range(N):
        roi_dens[i] = math.exp(-1 * torch.sum(dists[i, :] * s_dis[i, :]))

    return roi_dens


def get_deltas(dists, rho):
    N = dists.shape[0]
    deltas = torch.zeros(N)
    nearest_neiber = torch.zeros(N)
    index_rho = torch.argsort(-rho)

    for i, index in enumerate(index_rho):
        if i == 0:
            continue
        index_higher_rho = index_rho[:i]
        deltas[index] = torch.min(dists[index, index_higher_rho])
        index_nn = torch.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = torch.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        labs[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labs[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
            labs[index] = labs[int(nearest_neiber[index])]
    return labs


def draw_decision(rho, deltas, name="0_decision.jpg"):
    plt.cla()
    for i in range(np.shape(datas)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    plt.savefig(name)


def draw_cluster(datas, labs, centers, dic_colors, name="0_cluster.jpg"):
    plt.cla()
    K = np.shape(centers)[0]

    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit_transform(datas)

    for k in range(K):
        sub_index = torch.where(labs == k)
        sub_datas = x_tsne[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k])
        plt.scatter(x_tsne[centers[k], 0], x_tsne[centers[k], 1], color="k", marker="+", s=200.)
    plt.savefig(name)

def Spatial_DPC(datas):
    dists = getHighDistanceMatrix(datas)
    rho = get_density(dists, lambda1=50)
    deltas, nearest_neiber = get_deltas(dists, rho)
    centers = find_centers_K(rho, deltas, 3)
    return centers

# if __name__ == "__main__":
#     dic_colors = {0:(.8,0,0),1:(0,.8,0),
#                   2:(0,0,.8),3:(.8,.8,0),
#                   4:(.8,0,.8),5:(0,.8,.8),
#                   6:(0,0,0)}
#     file_name = "spatial"
#     with open(file_name+".txt","r",encoding="utf-8") as f:
#         lines = f.read().splitlines()
#     lines = [line.split("\t")[:-1] for line in lines]
#     datas = np.array(lines).astype(np.float32)
#     lambda1 = 50
#     dists = getHighDistanceMatrix(datas)
#     rho = get_density(dists, lambda1)
#     deltas, nearest_neiber = get_deltas(dists, rho)
#     draw_decision(rho, deltas, name=file_name+"_decision.jpg")
#     centers = find_centers_K(rho, deltas, 3)
#     print("centers", centers)
#     labs = cluster_PD(rho, centers, nearest_neiber)
#     draw_cluster(datas, labs, centers, dic_colors, name=file_name+"_cluster.jpg")



