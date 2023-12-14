import time
import math as M
import numpy as np
import Functions as f
import Color_Space_Change as csc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def show_as_gray(img: np.ndarray, imgTitle: str) -> None:
    """
    Show image as gray scale

    ## Parameters

    * `img` [np.ndarray] - Image to be shown
    """
    # Write log to log.txt to save min, max and mean
    with open("HDR_Output/log.txt", "a") as log:
        log.write(
            imgTitle
            + ": min = "
            + str(np.min(img))
            + ", max = "
            + str(np.max(img))
            + ", mean = "
            + str(np.mean(img))
            + "\n"
        )

    # Create image to show
    # imgShow = np.zeros_like(img)
    # # Transfer to float
    # imgShow = imgShow.astype(np.float32)
    # # Normalize image to 0~255
    # minVal, maxVal = np.min(img), np.max(img)
    # imgShow = (img - minVal) / (maxVal - minVal) * 255
    # # Show image
    # imgShow = Image.fromarray(np.uint8(imgShow))
    # imgShow.show(title=imgTitle)
    # Save image
    # imgShow.save("HDR_Output/" + imgTitle + ".png")

    # Draw histogram
    # plt.clf()
    # plt.hist(img.ravel(), bins=2000)
    # plt.title(imgTitle)
    # plt.savefig("HDR_Output/hist_" + imgTitle + ".png")
    # plt.clf()


def Bilateral_Filter():
    # 分別對 L 和 Y 做雙邊濾波
    Lbf = np.zeros_like(L)  # Lbf: L bilateral filter (L雙邊濾波後的結果)
    Ybf = np.zeros_like(Y)  # Ybf: Y bilateral filter (Y雙邊濾波後的結果)
    p = (block1 - 1) // 2  # p: 雙邊濾波器的半徑
    # 對影像做對稱填補，以避免邊界問題
    Lp = np.pad(L, (p, p), "symmetric")
    Yp = np.pad(Y, (p, p), "symmetric")
    for i in tqdm(range(p, rows + p)):
        for j in range(p, cols + p):
            # Ltemp, Ytemp: 雙邊濾波器的分子 / kb: 雙邊濾波器的分母
            Ltemp, Ytemp, kb = 0, 0, 0
            # 對 L(i,j) 做雙邊濾波的運算 -> Lbf(i,j)
            for m in range(-p, p + 1):
                for n in range(-p, p + 1):
                    # 1. c((m,n), (i,j)) 計算與中心點的幾何距離，並轉換成權重  [權重與 sigmad 有關，sigmad 越大，權重越小]
                    d = M.sqrt(M.pow(m, 2) + M.pow(n, 2))  # 計算幾何距離
                    c = M.exp((-1 / 2) * M.pow(d / sigmad, 2))  # 將距離轉換成權重
                    # 2. s((m,n), (i,j)) 計算與中心點的光度相似性，並轉換成權重  [權重與 sigmar 有關，sigmar 越大，權重越小]
                    delta = abs(Lp[i, j] - Lp[i + m, j + n])  # 計算光度相似性
                    s = M.exp((-1 / 2) * M.pow(delta / sigmar, 2))  # 將相似性轉換成權重
                    # 3. 將 c 與 s 的結果相乘，並統計起來做成分子、分母
                    Ltemp += L[i - p, j - p] * c * s  # 分子的部分（額外乘上 L(i,j)）
                    kb += c * s  # 分母的部分
            Lbf[i - p, j - p] = (1 / kb) * Ltemp  # 將分子除以分母，得到 Lbf(i,j) 的結果
            # 對 Y(i,j) 做雙邊濾波的運算 -> Ybf(i,j)
            kb = 0  # kb: 雙邊濾波器的分母（歸零）
            for m in range(-p, p + 1):
                for n in range(-p, p + 1):
                    # 1. c((m,n), (i,j)) 計算與中心點的幾何距離，並轉換成權重  [權重與 sigmad 有關，sigmad 越大，權重越小]
                    d = M.sqrt(M.pow(m, 2) + M.pow(n, 2))  # 計算幾何距離
                    c = M.exp((-1 / 2) * M.pow(d / sigmad, 2))  # 將距離轉換成權重
                    # 2. s((m,n), (i,j)) 計算與中心點的光度相似性，並轉換成權重  [權重與 sigmar 有關，sigmar 越大，權重越小]
                    delta = abs(Yp[i, j] - Yp[i + m, j + n])  # 計算光度相似性
                    s = M.exp((-1 / 2) * M.pow(delta / sigmar, 2))  # 將相似性轉換成權重
                    # 3. 將 c 與 s 的結果相乘，並統計起來做成分子、分母
                    Ytemp += Y[i - p, j - p] * c * s  # 分子的部分（額外乘上 Y(i,j)）
                    kb += c * s  # 分母的部分
            Ybf[i - p, j - p] = (1 / kb) * Ytemp  # 將分子除以分母，得到 Ybf(i,j) 的結果
    return Lbf, Ybf


def Local_Contrast_Extraction():
    p = (block2 - 1) // 2
    Lp = np.pad(Lbf, (p, p), "symmetric")
    Yp = np.pad(Ybf, (p, p), "symmetric")
    Llcme = np.zeros_like(Lbf)
    Ylcme = np.zeros_like(Ybf)
    for i in tqdm(range(p, rows + p)):
        for j in range(p, cols + p):
            Lavg = np.mean(Lp[i - p : i + p + 1, j - p : j + p + 1])
            Llcme[i - p, j - p] = np.log(Lbf[i - p, j - p] / (Lavg + bias1) + bias2)
            Yavg = np.mean(Yp[i - p : i + p + 1, j - p : j + p + 1])
            Ylcme[i - p, j - p] = np.log(Ybf[i - p, j - p] / (Yavg + bias1) + bias2)

    Llcme = (Llcme - np.min(Llcme)) / (np.max(Llcme) - np.min(Llcme))
    Ylcme = (Ylcme - np.min(Ylcme)) / (np.max(Ylcme) - np.min(Ylcme))
    return Llcme, Ylcme


def Histogram_Equalization():
    hL = np.zeros((Lbin))
    hY = np.zeros((Ybin))
    Llvl = rang / Lbin
    Ylvl = rang / Ybin
    Lindex = np.zeros((rows, cols), dtype=int)
    Yindex = np.zeros((rows, cols), dtype=int)
    for i in tqdm(range(rows)):
        for j in range(cols):
            for lvl in range(Lbin):
                if lvl * Llvl <= Llcme[i, j] and Llcme[i, j] < (lvl + 1) * Llvl:
                    hL[lvl] += 1
                    Lindex[i, j] = lvl
                    break
            for lvl in range(Ybin):
                if lvl * Ylvl <= Ylcme[i, j] and Ylcme[i, j] < (lvl + 1) * Ylvl:
                    hY[lvl] += 1
                    Yindex[i, j] = lvl
                    break
    plt.plot(hL, label="hL", linestyle="-")

    cfdL = np.zeros((Lbin))
    cfdY = np.zeros((Ybin))
    cfdL[0] = hL[0]
    cfdY[0] = hY[0]
    for lvl in range(1, Lbin):
        cfdL[lvl] = cfdL[lvl - 1] + hL[lvl]
    for lvl in range(1, Ybin):
        cfdY[lvl] = cfdY[lvl - 1] + hY[lvl]
    plt.plot(cfdL, label="cfdL", linestyle="--")

    nhL = np.zeros((Lbin))
    nhY = np.zeros((Ybin))
    Lhe = np.zeros_like(Llcme)
    Yhe = np.zeros_like(Ylcme)
    cfdLmin = np.min(cfdL)
    cfdLmax = np.max(cfdL)
    cfdYmin = np.min(cfdY)
    cfdYmax = np.max(cfdY)
    for i in tqdm(range(rows)):
        for j in range(cols):
            Lhe[i, j] = ((cfdL[Lindex[i, j]] - cfdLmin) / (cfdLmax - cfdLmin)) * rang
            for lvl in range(Lbin):
                if lvl * Llvl <= Lhe[i, j] and Lhe[i, j] < (lvl + 1) * Llvl:
                    nhL[lvl] += 1
                    break
            Yhe[i, j] = ((cfdY[Yindex[i, j]] - cfdYmin) / (cfdYmax - cfdYmin)) * rang
            for lvl in range(Lbin):
                if lvl * Ylvl <= Yhe[i, j] and Yhe[i, j] < (lvl + 1) * Ylvl:
                    nhY[lvl] += 1
                    break
    plt.plot(nhL, label="Lhe", linestyle="-.")
    plt.xticks()
    plt.yticks()
    return Lhe, Yhe


def Similar_Filter():
    p = (block3 - 1) // 2
    Lp = np.pad(Lhe, (p, p), "symmetric")
    Yp = np.pad(Yhe, (p, p), "symmetric")
    Lsf = Lhe.copy()
    Ysf = Yhe.copy()
    sL = np.zeros((block3, block3))
    sY = np.zeros((block3, block3))
    for i in tqdm(range(p, rows + p)):
        for j in range(p, cols + p):
            for m in range(-p, p + 1):
                for n in range(-p, p + 1):
                    if (
                        alpha1 * Lp[i + m, j + n] <= Lp[i, j]
                        and Lp[i, j] <= alpha2 * Lp[i + m, j + n]
                    ):
                        sL[p + m, p + n] = 1
                    else:
                        sL[p + m, p + n] = 0
            c = np.sum(sL)
            for m in range(-p, p + 1):
                for n in range(-p, p + 1):
                    Lsf[i - p, j - p] += (sL[p + m, p + n] * Lp[i + m, j + n]) / c

            for m in range(-p, p + 1):
                for n in range(-p, p + 1):
                    if (
                        alpha1 * Yp[i + m, j + n] <= Yp[i, j]
                        and Yp[i, j] <= alpha2 * Yp[i + m, j + n]
                    ):
                        sY[p + m, p + n] = 1
                    else:
                        sY[p + m, p + n] = 0
            c = np.sum(sY)
            for m in range(-p, p + 1):
                for n in range(-p, p + 1):
                    Ysf[i - p, j - p] += (sY[p + m, p + n] * Yp[i + m, j + n]) / c
    return Lsf, Ysf


def Y_Adjustment():
    La = np.zeros((rows, cols))
    Ya = np.zeros((rows, cols))
    for i in tqdm(range(rows)):
        for j in range(cols):
            La[i, j] = M.sqrt(Lsf[i, j] * L[i, j])
            Ya[i, j] = M.sqrt(Ysf[i, j] * Y[i, j])
    print(np.min(La), np.max(La), np.mean(La))
    return La, Ya


def Initial_Scaling():
    Lis = np.zeros((rows, cols))
    Yis = np.zeros((rows, cols))
    Ltemp = 0
    Ytemp = 0
    for i in tqdm(range(rows)):
        for j in range(cols):
            Ltemp += M.log(La[i, j] + delta1)
            Ytemp += M.log(Ya[i, j] + delta1)
    print(Ltemp)
    Lavg = M.exp(Ltemp / (rows * cols))
    Yavg = M.exp(Ytemp / (rows * cols))
    print(Lavg, Yavg)
    for i in tqdm(range(rows)):
        for j in range(cols):
            Lis[i, j] = alpha3 * (La[i, j] / Lavg)
            Yis[i, j] = alpha3 * (Ya[i, j] / Yavg)
    print(np.min(Lis), np.max(Lis), np.mean(Lis))
    return Lis, Yis


def Tone_mapping():
    Ltm = np.zeros((rows, cols))
    Ytm = np.zeros((rows, cols))
    Lw = M.exp(np.max(np.log(La + delta1)))  # np.exp(np.max(La))
    Yw = M.exp(np.max(np.log(Ya + delta1)))  # np.exp(np.max(Ya))
    print(Lw, Yw)
    for i in tqdm(range(rows)):
        for j in range(cols):
            Ltm[i, j] = Lis[i, j] * (1 + (Lis[i, j] / M.pow(Lw, 2))) / (1 + Lis[i, j])
            Ytm[i, j] = Yis[i, j] * (1 + (Yis[i, j] / M.pow(Yw, 2))) / (1 + Yis[i, j])
    # print('Ltm: ', np.min(Ltm), np.max(Ltm), np.mean(Ltm))
    # Lmax = np.max(Ltm)
    # Lmin = np.min(Ltm)
    # Ymax = np.max(Ytm)
    # Ymin = np.min(Ytm)
    # Ltm = (Ltm-Lmin)/(Lmax-Lmin)
    # Ytm = (Ytm-Ymin)/(Ymax-Ymin)
    # print('Ltm: ', np.min(Ltm), np.max(Ltm), np.mean(Ltm))
    return Ltm, Ytm


## Input
FileName = "Output/CCM_Process_F.png"
img = np.array(Image.open(FileName), dtype=np.float32)
# img = np.array(Image.open(FileName).convert('L'), dtype=np.float32)

## Parameter
rows, cols, _ = img.shape
# rows, cols = img.shape
Gimg = np.zeros((rows, cols, 3))
for chls in range(3):
    Gimg[:, :, chls] = img[:, :, chls]
    Lab = np.zeros_like(Gimg)
    Yxy = np.zeros_like(Gimg)
XYZ = np.zeros_like(Gimg)
for row in range(rows):
    for col in range(cols):
        Lab[row, col] = csc.sRGB_to_Lab(Gimg[row, col])
        GGimg = csc.sRGB_to_linearRGB(Gimg[row, col])
        XYZ[row, col, 0], XYZ[row, col, 1], XYZ[row, col, 2] = csc.RGB_to_XYZ(
            GGimg[0], GGimg[1], GGimg[2]
        )
        Yxy[row, col] = csc.XYZ_to_Yxy(
            XYZ[row, col, 0], XYZ[row, col, 1], XYZ[row, col, 2]
        )


L = Lab[:, :, 0]
Y = Yxy[:, :, 0]

# ? Debug - Show image
show_as_gray(L, "L0_origin")
show_as_gray(Y, "Y0_origin")

## Bilateral Filter
print("\nBilateral Filter:")
start = time.time()
block1 = 5
sigmad = 20
sigmar = 100
Lbf, Ybf = Bilateral_Filter()
f.time_elapsed(start)

# ? Debug - Show image
show_as_gray(Lbf, "L1_bilateral_filter")
show_as_gray(Ybf, "Y1_bilateral_filter")

## Local Contrast Map Evaluation
print("\nLocal Contrast Map Evaluation:")
start = time.time()
block2 = 5
bias1 = 1
bias2 = 1
Llcme, Ylcme = Local_Contrast_Extraction()
f.time_elapsed(start)

# ? Debug - Show image
show_as_gray(Llcme, "L2_local_contrast_ME")
show_as_gray(Ylcme, "Y2_local_contrast_ME")

## Histogram Equalization
print("\nHistogram Equalization:")
start = time.time()
Lbin = 100
Ybin = 100
rang = 1
Lhe, Yhe = Histogram_Equalization()
f.time_elapsed(start)

# ? Debug - Show image
show_as_gray(Lhe, "L3_histogram_equalization")
show_as_gray(Yhe, "Y3_histogram_equalization")

## Similar Filter
print("\nSimilar Filter:")
start = time.time()
block3 = 5
alpha1 = 0.9
alpha2 = 1.1
Lsf, Ysf = Similar_Filter()
f.time_elapsed(start)

show_as_gray(Lsf, "L4_similar_filter")
show_as_gray(Ysf, "Y4_similar_filter")

## Y Adjustment
print("\nY Adjustment:")
start = time.time()
La, Ya = Y_Adjustment()
f.time_elapsed(start)

# ? Debug - Show image
show_as_gray(La, "L5_Y_adjustment")
show_as_gray(Ya, "Y5_Y_adjustment")

## Initial Scaling
print("\nInitial Scaling:")
start = time.time()
alpha3 = 0.18
delta1 = 0.1
Lis, Yis = Initial_Scaling()
f.time_elapsed(start)

# ? Debug - Show image
show_as_gray(Lis, "L6_initial_scaling")
show_as_gray(Yis, "Y6_initial_scaling")

## Tone Mapping
print("\nTone Mapping:")
start = time.time()
Ltm, Ytm = Tone_mapping()
f.time_elapsed(start)

# ? Debug - Show image
show_as_gray(Ltm, "L7_tone_mapping")
show_as_gray(Ytm, "Y7_tone_mapping")

print(np.min(Ltm), np.max(Ltm))

## Output
out1 = np.zeros_like(Gimg)
out2 = np.zeros_like(Gimg)
Lab[:, :, 0] = Ltm
Yxy[:, :, 0] = Ytm

for row in range(rows):
    for col in range(cols):
        XYZ[row, col, 0], XYZ[row, col, 1], XYZ[row, col, 2] = csc.Yxy_to_XYZ(
            Yxy[row, col, 0], Yxy[row, col, 1], Yxy[row, col, 2]
        )


for row in range(rows):
    for col in range(cols):
        o1, o2, o3 = csc.Lab_to_XYZ(
            Lab[row, col, 0] * 100,
            Lab[row, col, 1] * 255 - 128,
            Lab[row, col, 2] * 255 - 128,
        )
        out1[row, col, 0], out1[row, col, 1], out1[row, col, 2] = csc.XYZ_to_RGB(
            o1, o2, o3
        )
        out2[row, col, 0], out2[row, col, 1], out2[row, col, 2] = csc.XYZ_to_RGB(
            XYZ[row, col, 0], XYZ[row, col, 1], XYZ[row, col, 2]
        )

out1 *= 255
Rmax = np.max(out1[:, :, 0])
Rmin = np.min(out1[:, :, 0])
Gmax = np.max(out1[:, :, 1])
Gmin = np.min(out1[:, :, 1])
Bmax = np.max(out1[:, :, 2])
Bmin = np.min(out1[:, :, 2])
out1[:, :, 0] = (out1[:, :, 0] - Rmin) / (Rmax - Rmin)
out1[:, :, 1] = (out1[:, :, 1] - Gmin) / (Gmax - Gmin)
out1[:, :, 2] = (out1[:, :, 2] - Bmin) / (Bmax - Bmin)

out2 *= 255
Rmax = np.max(out2[:, :, 0])
Rmin = np.min(out2[:, :, 0])
Gmax = np.max(out2[:, :, 1])
Gmin = np.min(out2[:, :, 1])
Bmax = np.max(out2[:, :, 2])
Bmin = np.min(out2[:, :, 2])
out2[:, :, 0] = (out2[:, :, 0] - Rmin) / (Rmax - Rmin)
out2[:, :, 1] = (out2[:, :, 1] - Gmin) / (Gmax - Gmin)
out2[:, :, 2] = (out2[:, :, 2] - Bmin) / (Bmax - Bmin)

for row in range(rows):
    for col in range(cols):
        out1[row, col] = csc.linearRGB_to_sRGB(out1[row, col] * 255)
        out2[row, col] = csc.linearRGB_to_sRGB(out2[row, col] * 255)

Gimg = np.clip(Gimg, 0, 255)
Gimg = Image.fromarray(np.uint8(Gimg))
# Gimg.show()
out1 = np.clip(out1, 0, 255)
out1 = Image.fromarray(np.uint8(out1))
out1.show()
out2 = np.clip(out2, 0, 255)
out2 = Image.fromarray(np.uint8(out2))
out2.show()

# Save image
out1.save("Output/CCMF_HDR_LAB.png")
out2.save("Output/CCMF_HDR_Yxy.png")
