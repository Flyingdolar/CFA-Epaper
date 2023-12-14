import math as M
import numpy as np
import matplotlib.pyplot as plt
import Color_Space_Change as csc
from PIL import Image
from tqdm import tqdm
from pyswarm import pso
from scipy.stats import linregress
from sklearn.cluster import KMeans
from matplotlib.patches import Circle

np.set_printoptions(suppress=True)

pcsO = np.zeros((1, 24, 3), dtype=np.float32)
pcsO[0] = [
    [116, 82, 68],  # No. 1
    [194, 150, 130],  # No. 2
    [98, 122, 157],  # No. 3
    [87, 108, 67],  # No. 4
    [175, 54, 60],  # No.15
    [255, 0, 0],  # Red
    [231, 199, 31],  # No.16
    [255, 255, 0],  # Yellow
    [214, 126, 44],  # No. 7
    [80, 91, 166],  # No. 8
    [193, 90, 99],  # No. 9
    [94, 60, 108],  # No.10
    [70, 148, 73],  # No.14
    [0, 255, 0],  # Green
    [8, 133, 161],  # No.18
    [0, 255, 255],  # Cyan
    [133, 128, 177],  # No. 5
    [103, 189, 170],  # No. 6
    [157, 188, 64],  # No.11
    [224, 163, 46],  # No.12
    [56, 61, 150],  # No.13
    [0, 0, 255],  # Blue
    [187, 86, 149],  # No.17
    [255, 0, 255],
]  # Magenta

# [255, 255, 255],  # White
# [243, 243, 242],  # No.19
# [200, 200, 200],  # No.20
# [160, 160, 160],  # No.21
# [122, 122, 121],  # No.22
# [ 85,  85,  85],  # No.23
# [ 52,  52,  52],  # No.24
# [  0,   0,   0]]  # Black

pcsM = np.zeros((1, 24, 3), dtype=np.float32)
pcsM[0] = [
    [13.66824532, 15.19980431, 15.78527355],  # No. 1
    [24.67165947, 27.64808655, 27.75913811],  # No. 2
    [16.41493988, 22.11903191, 23.85330772],  # No. 3
    [12.19178391, 15.26957130, 15.23969269],  # No. 4
    [16.32714653, 15.03137207, 16.09106445],  # No.15
    [19.32901001, 11.55959606, 12.52017498],  # Red
    [24.84242439, 25.44982529, 21.78178215],  # No. 16
    [30.24589920, 30.65556717, 23.80205154],  # Yellow
    [22.40748596, 21.55843544, 20.33401299],  # No. 7
    [13.90341759, 19.06546593, 21.61734390],  # No. 8
    [20.07892036, 20.09754753, 21.18385887],  # No. 9
    [11.70565319, 14.07978630, 15.89845562],  # No.10
    [12.52267361, 17.44430923, 16.51893616],  # No.14
    [10.11330509, 19.11426353, 13.14266777],  # Green
    [9.241112709, 16.75541878, 18.15682411],  # No.18
    [17.14426804, 33.16486740, 32.27040482],  # Cyan
    [19.34404182, 24.51961136, 26.55222130],  # No. 5
    [19.92150307, 28.28966904, 28.30182838],  # No. 6
    [19.99352264, 23.93764687, 21.73658752],  # No.11
    [23.44618607, 23.61075401, 21.49836731],  # No.12
    [10.32926178, 14.32370758, 17.10609818],  # No.13
    [7.340748787, 13.32271099, 19.77804565],  # Blue
    [20.14383698, 21.27284813, 23.74345970],  # No.17
    [27.35544777, 24.73938179, 31.49990273],
]  # Magenta

# [39.69083023,	46.14099121,	44.42738724],   # White
# [35.72702789,	42.24011993,	41.31967926],   # No.19
# [28.87051964,	35.17556381,	35.45739746],   # No.20
# [22.02905846,	26.96634865,	27.63151169],   # No.21
# [15.35199261,	18.85037422,	19.68402290],   # No.22
# [10.66071129,	13.15429115,	14.13704681],   # No.23
# [7.933727741,	9.694092751,	10.58777714],   # No.24
# [3.949607134,	4.656770706,	5.483669281]]   # Black


def Auto_White_Balance():
    return 0


def Auto_Level(From, To):
    XYZf = np.zeros_like(From)
    XYZt = np.zeros_like(To)
    Labf = np.zeros_like(From)
    Labt = np.zeros_like(To)
    for clr in range(From.shape[1]):
        XYZf[0, clr] = csc.sRGB_to_linearRGB(From[0, clr])
        XYZf[0, clr, 0], XYZf[0, clr, 1], XYZf[0, clr, 2] = csc.RGB_to_XYZ(
            XYZf[0, clr, 0], XYZf[0, clr, 1], XYZf[0, clr, 2]
        )
        XYZt[0, clr] = csc.sRGB_to_linearRGB(To[0, clr])
        XYZt[0, clr, 0], XYZt[0, clr, 1], XYZt[0, clr, 2] = csc.RGB_to_XYZ(
            XYZt[0, clr, 0], XYZt[0, clr, 1], XYZt[0, clr, 2]
        )

        Labf[0, clr] = csc.sRGB_to_Lab(From[0, clr])
        Labt[0, clr] = csc.sRGB_to_Lab(To[0, clr])

    XYZf[:, :, 1] = XYZt[:, :, 1]
    Labf[:, :, 0] = Labt[:, :, 0]

    pcsX = np.zeros_like(From)
    pcsL = np.zeros_like(From)
    for clr in range(pcsX.shape[1]):
        pcsX[0, clr, 0], pcsX[0, clr, 1], pcsX[0, clr, 2] = csc.XYZ_to_RGB(
            XYZf[0, clr, 0], XYZf[0, clr, 1], XYZf[0, clr, 2]
        )
        pcsX[0, clr] = csc.linearRGB_to_sRGB(pcsX[0, clr])

        pcsL[0, clr] = csc.Lab_to_sRGB(Labf[0, clr])

    return XYZf, XYZt, pcsX, Labf, Labt, pcsL


def Train(x, pcs, Lab):
    ccm = np.array(
        [
            [1 - x[0] - x[1], x[0], x[1]],
            [x[2], 1 - x[2] - x[3], x[3]],
            [x[4], x[5], 1 - x[4] - x[5]],
        ]
    )

    pcsCCM = np.zeros_like(pcs)
    for clr in range(pcs.shape[1]):
        pcsCCM[0, clr, 0] = (
            ccm[0, 0] * pcs[0, clr, 0]
            + ccm[0, 1] * pcs[0, clr, 1]
            + ccm[0, 2] * pcs[0, clr, 2]
        )
        pcsCCM[0, clr, 1] = (
            ccm[1, 0] * pcs[0, clr, 0]
            + ccm[1, 1] * pcs[0, clr, 1]
            + ccm[1, 2] * pcs[0, clr, 2]
        )
        pcsCCM[0, clr, 2] = (
            ccm[2, 0] * pcs[0, clr, 0]
            + ccm[2, 1] * pcs[0, clr, 1]
            + ccm[2, 2] * pcs[0, clr, 2]
        )

    LabCCM = np.zeros_like(pcs)
    for clr in range(pcs.shape[1]):
        LabCCM[0, clr] = csc.sRGB_to_Lab(pcsCCM[0, clr])

    temp = 0
    Emax = 0
    Eavg = 0
    for clr in range(pcs.shape[1]):
        temp = M.sqrt(np.sum(np.power(LabCCM[0, clr] - Lab[0, clr], 2)))
        Eavg += temp
        if temp > Emax:
            Emax = temp
    Eavg /= pcs.shape[1]

    print("Average Loss:", Eavg, "\tMax Loss:", Emax)
    return Eavg


def Condition(x, _1, _2):
    return [1 - x[0] - x[1], 1 - x[2] - x[3], 1 - x[4] - x[5]]


def Particle_Swarm_Optimization(weight, limit, pcs, Lab, index):
    lb = []
    ub = []
    for i in range(weight):
        lb.append(-limit)
        ub.append(limit)

    best, _ = pso(
        Train, lb, ub, args=(pcs, Lab), ieqcons=[Condition], swarmsize=24, maxiter=1000
    )
    CCMbest = np.array(
        [
            [1 - best[0] - best[1], best[0], best[1]],
            [best[2], 1 - best[2] - best[3], best[3]],
            [best[4], best[5], 1 - best[4] - best[5]],
        ]
    )

    pcsCCM = np.zeros_like(pcs)
    for clr in range(pcs.shape[1]):
        pcsCCM[0, clr, 0] = (
            CCMbest[0, 0] * pcs[0, clr, 0]
            + CCMbest[0, 1] * pcs[0, clr, 1]
            + CCMbest[0, 2] * pcs[0, clr, 2]
        )
        pcsCCM[0, clr, 1] = (
            CCMbest[1, 0] * pcs[0, clr, 0]
            + CCMbest[1, 1] * pcs[0, clr, 1]
            + CCMbest[1, 2] * pcs[0, clr, 2]
        )
        pcsCCM[0, clr, 2] = (
            CCMbest[2, 0] * pcs[0, clr, 0]
            + CCMbest[2, 1] * pcs[0, clr, 1]
            + CCMbest[2, 2] * pcs[0, clr, 2]
        )

    XYZccm = np.zeros_like(pcsCCM)
    Labccm = np.zeros_like(pcsCCM)
    for clr in range(pcs.shape[1]):
        XYZccm[0, clr] = csc.sRGB_to_linearRGB(pcsCCM[0, clr])
        XYZccm[0, clr, 0], XYZccm[0, clr, 1], XYZccm[0, clr, 2] = csc.RGB_to_XYZ(
            XYZccm[0, clr, 0], XYZccm[0, clr, 1], XYZccm[0, clr, 2]
        )

        Labccm[0, clr] = csc.sRGB_to_Lab(pcsCCM[0, clr])

    c = 0
    if index:
        for clr in range(pcsO.shape[1]):
            if labels[clr] == index - 1:
                XYZccm[0, c, 1] = XYZo[0, clr, 1]
                Labccm[0, c, 0] = Labo[0, clr, 0]
                c += 1
    else:
        XYZccm[:, :, 1] = XYZo[:, :, 1]
        Labccm[:, :, 0] = Labo[:, :, 0]

    pcsCCML = np.zeros_like(pcsCCM)
    for clr in range(pcs.shape[1]):
        pcsCCML[0, clr] = csc.Lab_to_sRGB(Labccm[0, clr])
    pcsCCML = np.clip(pcsCCML, 0, 255)
    return pcsCCML, Labccm, CCMbest


def Color_Checker(pcs):
    bgcolor = [0, 141, 141]
    rows, cols, chls = 1404, 1872, 3
    rClrs, cClrs = 3, 8
    block, margin = 200, 30
    Is = int((rows - rClrs * block - (rClrs - 1) * margin) / 2)
    Js = int((cols - cClrs * block - (cClrs - 1) * margin) / 2)
    out = np.full((rows, cols, chls), bgcolor[0])
    clrs = 0

    for i in tqdm(range(Is, rows - Is - block + 1, block + margin)):
        for j in range(Js, cols - Js - block + 1, block + margin):
            for x in range(block):
                for y in range(block):
                    out[i + x, j + y] = np.round(pcs[0, clrs], decimals=1)
            clrs += 1

    out = np.clip(out, 0, 255)
    out = Image.fromarray(np.uint8(out))
    out.show()


def Plot(pcs, Lab, draw):
    clrs = [
        (pcs[0, clr, 0] / 255, pcs[0, clr, 1] / 255, pcs[0, clr, 2] / 255, 1)
        for clr in range(pcs.shape[1])
    ]
    a = [Lab[0, clr, 1] for clr in range(Lab.shape[1])]
    b = [Lab[0, clr, 2] for clr in range(Lab.shape[1])]
    a = np.array(a) * 255 - 128
    b = np.array(b) * 255 - 128
    if draw:
        plt.scatter(a, b, c=clrs, marker="o", s=500, cmap="viridis", label="Colors")
        plt.xlabel("a*")
        plt.ylabel("b*")
        plt.title("L*a*b* Color Space")
        for i, txt in enumerate(range(1, Lab.shape[1] + 1)):
            plt.text(
                a[i] + 0.6, b[i] + 2, str(txt), fontsize=15, ha="right", va="bottom"
            )
        plt.show()

    XYZ = np.zeros_like(Lab)
    for clr in range(XYZ.shape[1]):
        XYZ[0, clr, 0], XYZ[0, clr, 1], XYZ[0, clr, 2] = csc.Lab_to_XYZ(
            Lab[0, clr, 0] * 100, Lab[0, clr, 1] * 255 - 128, Lab[0, clr, 2] * 255 - 128
        )
    x = [XYZ[0, clr, 0] / (np.sum(XYZ[0, clr])) for clr in range(XYZ.shape[1])]
    y = [XYZ[0, clr, 1] / (np.sum(XYZ[0, clr])) for clr in range(XYZ.shape[1])]
    if draw:
        plt.scatter(x, y, c=clrs, marker="o", s=500, cmap="viridis", label="Colors")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Yxy Color Space")
        for i, txt in enumerate(range(1, Lab.shape[1] + 1)):
            plt.text(x[i], y[i], str(txt), fontsize=15, ha="right", va="bottom")
        plt.show()

    return a, b, x, y, clrs


def K_means_Cluster(x, y, draw):
    target = np.zeros((pcsCCML.shape[1], 2))
    for i in range(target.shape[0]):
        target[i] = [x[i], y[i]]

    cluster = 3
    kmeans = KMeans(n_clusters=cluster, random_state=42)
    kmeans.fit(target)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    if draw:
        plt.scatter(
            target[:, 0], target[:, 1], c=clrs, cmap="viridis", marker="o", s=100
        )
        plt.scatter(
            centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids"
        )
        for i in range(cluster):
            cluster_points = target[labels == i]
            center = centers[i]
            radius = np.max(np.linalg.norm(cluster_points - center, axis=1))

            circle = Circle(
                center, radius, linewidth=1, edgecolor="green", facecolor="none"
            )
            plt.gca().add_patch(circle)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("K-means Clustering")
        plt.legend()
        plt.show()

    return labels, centers


## Auto White Balance
Auto_White_Balance()

## Auto Level
XYZl, XYZo, pcsX, Labl, Labo, pcsL = Auto_Level(pcsM, pcsO)

## First Step Particle Swarm Optimization
limit0 = 4.0
pcsCCML, LabCCML, ccm0 = Particle_Swarm_Optimization(6, limit0, pcsL, Labo, 0)
Color_Checker(pcsCCML)

## K-means Cluster
draw = True
a, b, x, y, clrs = Plot(pcsCCML, LabCCML, draw)
labels, centers = K_means_Cluster(a, b, draw)

## Second Step Particle Swarm Optimization
pcs1, pcs2, pcs3 = [], [], []
Labo1, Labo2, Labo3 = [], [], []
for i in range(len(labels)):
    if labels[i] == 0:
        pcs1.append(pcsL[0, i])
        Labo1.append(Labo[0, i])
    elif labels[i] == 1:
        pcs2.append(pcsL[0, i])
        Labo2.append(Labo[0, i])
    elif labels[i] == 2:
        pcs3.append(pcsL[0, i])
        Labo3.append(Labo[0, i])
    else:
        print("Error")
pcs1, pcs2, pcs3 = np.array([pcs1]), np.array([pcs2]), np.array([pcs3])
Labo1, Labo2, Labo3 = np.array([Labo1]), np.array([Labo2]), np.array([Labo3])

limit1 = 4.0
pcsCCM1, LabCCM1, ccm1 = Particle_Swarm_Optimization(6, limit1, pcs1, Labo1, 1)

limit2 = 4.0
pcsCCM2, LabCCM2, ccm2 = Particle_Swarm_Optimization(6, limit2, pcs2, Labo2, 2)

limit3 = 4.0
pcsCCM3, LabCCM3, ccm3 = Particle_Swarm_Optimization(6, limit3, pcs3, Labo3, 3)

pcsFCCM, LabFCCM = [], []
c1, c2, c3 = 0, 0, 0
for i in range(len(labels)):
    if labels[i] == 0:
        pcsFCCM.append(pcsCCM1[0, c1])
        LabFCCM.append(LabCCM1[0, c1])
        c1 += 1
    elif labels[i] == 1:
        pcsFCCM.append(pcsCCM2[0, c2])
        LabFCCM.append(LabCCM2[0, c2])
        c2 += 1
    elif labels[i] == 2:
        pcsFCCM.append(pcsCCM3[0, c3])
        LabFCCM.append(LabCCM3[0, c3])
        c3 += 1
    else:
        print("Error")
pcsFCCM, LabFCCM = np.array([pcsFCCM]), np.array([LabFCCM])
draw = True

ccmlist = [ccm0, ccm1, ccm2, ccm3]
for i in range(len(ccmlist)):
    print(f"Optimal Color Correction Matrix{i}:")
    print(ccmlist[i], "\n")

Color_Checker(pcsFCCM)
Plot(pcsFCCM, LabFCCM, draw)
