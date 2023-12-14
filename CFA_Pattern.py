import os
import time
import math as m
import numpy as np
import pandas as pd
import Functions as f
from PIL import Image
from scipy.signal import convolve2d, correlate2d
import os
import time
import math as m
import numpy as np
import pandas as pd
import Functions as f
from PIL import Image
from scipy.signal import convolve2d, correlate2d


def ToggleB(gm, Cpp, Cpe, hCpp, i, j, convergence):
    # if gm[i, j] < 0.5:
    #     if Cpe[i, j] < -0.5*Cpp[hCpp, hCpp]:
    #         for x in range(-hCpp, hCpp+1):
    #             for y in range(-hCpp, hCpp+1):
    #                 Cpe[i+x+hCpp, j+y+hCpp] += Cpp[x+hCpp, y+hCpp]
    #         gm[i, j] = 1
    #         convergence += 0
    # else:
    #     if Cpe[i, j] > 0.5*Cpp[hCpp, hCpp]:
    #         for x in range(-hCpp, hCpp+1):
    #             for y in range(-hCpp, hCpp+1):
    #                 Cpe[i+x+hCpp, j+y+hCpp] -= Cpp[x+hCpp, y+hCpp]
    #         gm[i, j] = 0
    #         convergence += 1

    amc = 0
    Delta_Emin = 0
    if np.isclose(gm[i, j], 1.0):
        am = -1
    else:
        am = 1

    Delta_E = am * am * Cpp[hCpp, hCpp] + 2 * am * Cpe[i + hCpp, j + hCpp]

    if Delta_Emin > Delta_E:
        Delta_Emin = Delta_E
        amc = am

    if Delta_Emin < 0:
        for x in range(-hCpp, hCpp + 1):
            for y in range(-hCpp, hCpp + 1):
                Cpe[i + x + hCpp, j + y + hCpp] += amc * Cpp[x + hCpp, y + hCpp]

        gm[i, j] += amc
        convergence += 1

    return gm, Cpe, convergence


def RTBDBS(fm, gm, sigma, size, imax):
    fm = np.array(fm, dtype=np.float32) / 255
    gm = np.array(gm, dtype=np.float32) / 255
    rows, cols = fm.shape

    sig = 18
    seq1616 = np.load("Input/Seq16x16.npy")
    # seq1818 = np.load("Input/Seq18x18.npy")
    # seq1820 = np.load("Input/data1.npy")
    # seq2018 = np.load("Input/data1.npy")
    seq2020 = np.load("Input/Seq20x20.npy")

    gs = (size + 1) // 2
    length = (gs - 1) // 2
    Gf = np.zeros((gs, gs))
    for x in range(-length, length + 1):
        for y in range(-length, length + 1):
            c = (m.pow(x, 2) + m.pow(y, 2)) / (2 * m.pow(sigma, 2))
            Gf[x + length, y + length] = m.exp(-c) / (2 * m.pi * m.pow(sigma, 2))
    hCpp = (size - 1) // 2
    Cpp = convolve2d(Gf, Gf, mode="full")
    e = gm - fm
    Cpe = correlate2d(e, Cpp, mode="full")

    # hCpp = 2
    # Cpp = np.zeros((2*hCpp+1, 2*hCpp+1), dtype=np.float32)
    # for i in range(2*hCpp+1):
    #     for j in range(2*hCpp+1):
    #         Cpp[i, j] = m.exp(-(m.pow(i - hCpp/2, 2) + m.pow(j - hCpp/2, 2))/(2*m.pow(sigma, 2)))

    # f0 = fm.copy()
    # g0 = gm.copy()
    # for i in range(hCpp, rows-hCpp):
    #     for j in range(hCpp, cols-hCpp):
    #         for x in range(-hCpp, hCpp+1):
    #             for y in range(-hCpp, hCpp+1):
    #                 f0[i, j] += fm[i+x, j+y]*Cpp[x+hCpp, y+hCpp]
    #                 g0[i, j] += gm[i+x, j+y]*Cpp[x+hCpp, y+hCpp]
    # Cpe = g0 - f0

    iteration = 0
    while True:
        iteration += 1
        convergence = 0

        for i in range(hCpp, rows - 15, 16):
            for j in range(hCpp, cols - 15, 16):
                for s in range(len(seq1616)):
                    k = seq1616[s, 0]
                    l = seq1616[s, 1]
                    gm, Cpe, convergence = ToggleB(
                        gm, Cpp, Cpe, hCpp, k + i, l + j, convergence
                    )

        # for i in range(hCpp, rows-15, 16):
        #     for j in range(hCpp, cols-15, 16):
        #         begin = np.random.randint(0, 255)
        #         for s in range(begin, 256):
        #             k = seq[s]//16
        #             l = seq[s]%16
        #             gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i+k, j+l, convergence)
        #         for s in range(begin):
        #             k = seq[s]//16
        #             l = seq[s]%16
        #             gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i+k, j+l, convergence)
        # for i in range(rows):
        #     if i < (rows - rows%16):
        #         for j in range(cols - cols%16, cols):
        #             gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i, j, convergence)
        #     else:
        #         for j in range(cols):
        #             gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i, j, convergence)

        print(iteration)
        if convergence == 0 or iteration >= imax:
            break

    gm = np.clip(gm * 255, 0, 255)
    return Image.fromarray(np.uint8(gm)), convergence


def ToggleC(gm, pcs, Cpp, Cpe, hCpp, i, j, convergence):
    amLc, amac, ambc = 0, 0, 0
    Delta_Emin = 0
    for cl in range(pcs.shape[1]):
        amL = pcs[0, cl, 0] - gm[i, j, 0]
        ama = pcs[0, cl, 1] - gm[i, j, 1]
        amb = pcs[0, cl, 2] - gm[i, j, 2]

        Delta_E = (
            amL * amL * Cpp[hCpp, hCpp]
            + 2 * amL * Cpe[i + hCpp, j + hCpp, 0]
            + ama * ama * Cpp[hCpp, hCpp]
            + 2 * ama * Cpe[i + hCpp, j + hCpp, 1]
            + amb * amb * Cpp[hCpp, hCpp]
            + 2 * amb * Cpe[i + hCpp, j + hCpp, 2]
        )

        if Delta_Emin > Delta_E:
            Delta_Emin = Delta_E
            amLc, amac, ambc = amL, ama, amb
        # elif Delta_Emin == Delta_E:
        #     print("Equal")

    if Delta_Emin < 0:
        for x in range(-hCpp, hCpp + 1):
            for y in range(-hCpp, hCpp + 1):
                Cpe[i + x + hCpp, j + y + hCpp, 0] += amLc * Cpp[x + hCpp, y + hCpp]
                Cpe[i + x + hCpp, j + y + hCpp, 1] += amac * Cpp[x + hCpp, y + hCpp]
                Cpe[i + x + hCpp, j + y + hCpp, 2] += ambc * Cpp[x + hCpp, y + hCpp]
        gm[i, j, 0] += amLc
        gm[i, j, 1] += amac
        gm[i, j, 2] += ambc
        convergence += 1
    return gm, Cpe, convergence


def RTBCDBS_20x20(fm, gm, pcs, sigma, size, imax):
    fm = np.array(fm, dtype=np.float32)
    gm = np.array(gm, dtype=np.float32)
    rows, cols, chls = fm.shape

    # seq1616 = np.load("Input/Seq16x16.npy")
    # seq1818 = np.load("Input/Seq18x18.npy")
    # seq1820 = np.load("Input/Seq18x20.npy")
    # seq2018 = np.load("Input/Seq20x18.npy")
    seq2020 = np.load("Input/Seq20x20.npy")

    # Pad
    p = 2
    fmp = np.zeros((rows + 2 * p, cols + 2 * p, chls))
    gmp = np.zeros((rows + 2 * p, cols + 2 * p, chls))
    for cl in range(chls):
        fmp[:, :, cl] = np.pad(fm[:, :, cl], (p, p), "symmetric")
        gmp[:, :, cl] = np.pad(gm[:, :, cl], (p, p), "symmetric")

    gs = (size + 1) // 2
    length = (gs - 1) // 2
    Gf = np.zeros((gs, gs))
    for x in range(-length, length + 1):
        for y in range(-length, length + 1):
            c = (m.pow(x, 2) + m.pow(y, 2)) / (2 * m.pow(sigma, 2))
            Gf[x + length, y + length] = m.exp(-c) / (2 * m.pi * m.pow(sigma, 2))
    hCpp = (size - 1) // 2
    Cpp = convolve2d(Gf, Gf, mode="full")

    # e = gm - fm
    # Cpe = np.zeros((rows + hCpp*2, cols + hCpp*2, chls))
    e = gmp - fmp
    Cpe = np.zeros((rows + p * 2 + hCpp * 2, cols + p * 2 + hCpp * 2, chls))
    for c in range(chls):
        Cpe[:, :, c] = correlate2d(e[:, :, c], Cpp, mode="full")

    iteration = 0
    while True:
        iteration += 1
        convergence = 0

        for i in range(0, rows - rows % 16, 16):
            for j in range(0, cols - cols % 16, 16):
                # if i < 2 or i+15 >= rows-2 or j < 2 or j+15 >= cols-2:
                #     for s in range(len(seq1616)):
                #         k = seq1616[s, 0]
                #         l = seq1616[s, 1]
                #         gm, Cpe, convergence = ToggleC(gm, pcs, Cpp, Cpe, hCpp, i+k, j+l, convergence)
                # else:
                for s in range(len(seq2020)):
                    k = seq2020[s, 0]
                    l = seq2020[s, 1]
                    gmp, Cpe, convergence = ToggleC(
                        gmp, pcs, Cpp, Cpe, hCpp, i - 2 + k, j - 2 + l, convergence
                    )
        for i in range(rows):
            if i < (rows - rows % 16):
                for j in range(cols - cols % 16, cols):
                    gmp, Cpe, convergence = ToggleC(
                        gmp, pcs, Cpp, Cpe, hCpp, i, j, convergence
                    )
            else:
                for j in range(cols):
                    gmp, Cpe, convergence = ToggleC(
                        gmp, pcs, Cpp, Cpe, hCpp, i, j, convergence
                    )

        if convergence == 0 or iteration >= imax:
            break

    # gm = np.clip(gm, 0, 255)
    out1 = gmp[p : rows + p, p : cols + p]
    return Image.fromarray(np.uint8(out1))


def ToggleStep(gm, cfa, Cpp, Cpe, hCpp, i, j, convergence):
    dm_ = 0
    Delta_Emin = 0
    for cl in range(step):
        dm = cl - gm[i, j, cfa]
        Delta_E = dm * dm * Cpp[hCpp, hCpp] + 2 * dm * Cpe[i + hCpp, j + hCpp, cfa]

        if Delta_Emin > Delta_E:
            Delta_Emin = Delta_E
            dm_ = dm

    if Delta_Emin < 0:
        for x in range(-hCpp, hCpp + 1):
            for y in range(-hCpp, hCpp + 1):
                Cpe[i + x + hCpp, j + y + hCpp, cfa] += dm_ * Cpp[x + hCpp, y + hCpp]
        gm[i, j, cfa] += dm_
        convergence += 1
    return gm, Cpe, convergence


def RTBDBS_CFA(fm, gm, step, sigma, size, imax):
    fm = (np.array(fm, dtype=np.float32) / 255) * (step - 1)
    # gm = np.array(gm, dtype=np.float32)
    gm = np.zeros((fm.shape))
    rows, cols, chls = fm.shape

    # CFA Pattern
    CFA = np.zeros((rows, cols))
    period = [2, 2, 0, 0, 1, 1]
    for i in range(rows):
        if i % 2 == 0:
            CFA[i, 0] = 1
            for j in range(1, cols):
                CFA[i, j] = period[(j - 1) % 6]
        else:
            for j in range(cols):
                CFA[i, j] = period[(j + 2) % 6]
    # CFA Fm
    CFA_fm = np.zeros((rows, cols, chls))
    for i in range(rows):
        for j in range(cols):
            for c in range(chls):
                if c == CFA[i, j]:
                    CFA_fm[i, j, c] = fm[i, j, c]
    # Pad
    p = 2
    gmp = np.zeros((rows + 2 * p, cols + 2 * p, chls))
    CFA_fmp = np.zeros((rows + 2 * p, cols + 2 * p, chls))
    CFAp = np.uint8(np.pad(CFA, (p, p), "symmetric"))
    for cl in range(chls):
        # fmp[:, :, cl] = np.pad(fm[:, :, cl], (p, p), 'symmetric')
        gmp[:, :, cl] = np.pad(gm[:, :, cl], (p, p), "symmetric")
        CFA_fmp[:, :, cl] = np.pad(CFA_fm[:, :, cl], (p, p), "symmetric")

    seq2020 = np.load("Input/Seq20x20.npy")
    gs = (size + 1) // 2
    length = (gs - 1) // 2
    Gf = np.zeros((gs, gs))
    for x in range(-length, length + 1):
        for y in range(-length, length + 1):
            c = (m.pow(x, 2) + m.pow(y, 2)) / (2 * m.pow(sigma, 2))
            Gf[x + length, y + length] = m.exp(-c) / (2 * m.pi * m.pow(sigma, 2))
    hCpp = (size - 1) // 2
    Cpp = convolve2d(Gf, Gf, mode="full")

    e = gmp - CFA_fmp
    Cpe = np.zeros((rows + p * 2 + hCpp * 2, cols + p * 2 + hCpp * 2, chls))
    for cl in range(chls):
        Cpe[:, :, cl] = correlate2d(e[:, :, cl], Cpp, mode="full")

    iteration = 0
    while True:
        iteration += 1
        convergence = 0

        for i in range(2, rows - rows % 16 + 2, 16):
            for j in range(2, cols - cols % 16 + 2, 16):
                begin = np.random.randint(0, 400)
                for s in range(begin, len(seq2020)):
                    k = seq2020[s, 0]
                    l = seq2020[s, 1]
                    gmp, Cpe, convergence = ToggleStep(
                        gmp,
                        CFAp[i - 2 + k, j - 2 + l],
                        Cpp,
                        Cpe,
                        hCpp,
                        i - 2 + k,
                        j - 2 + l,
                        convergence,
                    )
                for s in range(begin):
                    k = seq2020[s, 0]
                    l = seq2020[s, 1]
                    gmp, Cpe, convergence = ToggleStep(
                        gmp,
                        CFAp[i - 2 + k, j - 2 + l],
                        Cpp,
                        Cpe,
                        hCpp,
                        i - 2 + k,
                        j - 2 + l,
                        convergence,
                    )

        for i in range(rows):
            if i < (rows - rows % 16):
                for j in range(cols - cols % 16, cols):
                    gmp, Cpe, convergence = ToggleStep(
                        gmp, CFAp[i, j], Cpp, Cpe, hCpp, i, j, convergence
                    )
            else:
                for j in range(cols):
                    gmp, Cpe, convergence = ToggleStep(
                        gmp, CFAp[i, j], Cpp, Cpe, hCpp, i, j, convergence
                    )

        if convergence == 0 or iteration >= imax:
            break

    gmp = np.clip((gmp / (step - 1)) * 255, 0, 255)
    out1 = gmp[p : rows + p, p : cols + p]
    out2 = np.clip((CFA_fm / (step - 1)) * 255, 0, 255)
    return Image.fromarray(np.uint8(out1)), Image.fromarray(np.uint8(out2))



def apply_CFA():
    pcs = np.zeros((1, 8, 3), dtype=np.float32)
    pcs[0] = [
        [0, 0, 0],  # Black
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 255, 255],  # White
    ]

    fm = Image.open("Output/CCM_Process_R.png")
    g0 = f.random_halftone(fm, False, pcs)

    step = 16
    fr, fg, fb = fm.split()
    gr, gg, gb = g0.split()
    start = time.time()
    img, CFA_img = RTBDBS_CFA(fm, g0, step, 1, 13, 20)
    f.time_elapsed(start)
    bit = 8
    MAX_I = m.pow(2, bit) - 1
    MSE = f.Calculate_MSE(CFA_img, img, 5, 0.9)
    HPSNR = 10 * m.log10(m.pow(MAX_I, 2) / MSE)
    print(HPSNR)
    img.show()
    img.save("Output/CCMR_HT.png")
    CFA_img.show()
    CFA_img.save("Output/CCMR_CFA.png")

def RTBDBS_CFA(fm, gm, step, sigma, size, imax):
    # Implementation of RTBDBS_CFA function goes here

# process_image()
