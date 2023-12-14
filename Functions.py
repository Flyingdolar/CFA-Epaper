import time
import math as m
import numpy as np
import Color_Space_Change as csc
from scipy.signal import convolve2d, correlate2d
from PIL import Image

# seq = np.load("Input/data2.npy")
# np.savetxt('Sequence2.csv', seq, delimiter=',')

def time_elapsed(start):
    end = time.time()
    elapsed = end - start
    mins, secs = divmod(elapsed, 60.0)
    secs = round(secs * 10) // 10
    print("Elapsed time:", mins, "mins ", secs, "secs")
    return elapsed

### Evaluation ###
def Calculate_MSE(Fimg, Gimg, size, sigma):
    Fimg = np.array(Fimg, dtype=np.float32)
    Gimg = np.array(Gimg, dtype=np.float32)
    Gf = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            Gf[i, j] = m.exp(-(m.pow(i - size/2, 2) + m.pow(j - size/2, 2)) / (2*m.pow(sigma, 2)))

    Sum = 0
    for chnl in range(3):
        for i in range(0, Fimg.shape[0], size):
            for j in range(0, Fimg.shape[1], size):
                Segment = 0
                for x in range(size):
                    if i+x >= Fimg.shape[0]:
                        continue
                    for y in range(size):
                        if j+y >= Fimg.shape[1]:
                            continue
                        Segment += Gf[x, y]*(Fimg[i+x, j+y, chnl]-Gimg[i+x, j+y, chnl])
                Sum += m.pow(Segment, 2)
    return Sum/(3*Fimg.shape[0]*Fimg.shape[1])

### Random Halftone Image (First gm) ###
def random_halftone(img, color, pcs):
    if color:
        gmC = np.array(img, dtype=np.float32)
        for i in range(gmC.shape[0]):
            for j in range(gmC.shape[1]):
                color = np.random.randint(0, pcs.shape[1])
                gmC[i, j] = pcs[0][color]
        return Image.fromarray(np.uint8(gmC))
    else:
        gm = np.array(img, dtype=np.float32)
        for i in range(gm.shape[0]):
            for j in range(gm.shape[1]):
                for k in range(3):
                    gm[i, j, k] = np.random.randint(0, 2)*255
        return Image.fromarray(np.uint8(gm))

### Vector Error Diffusion ###
def distance(a,b):
    dist = m.sqrt(m.pow(a[0] - b[0], 2) + m.pow(a[1] - b[1], 2) + m.pow(a[2] - b[2], 2))
    return dist

def vector_error_diffusion(img, pcs):
    ## Color Space Change of Primary Colors
    vl_Lab = np.copy(pcs)
    for n in range(pcs.shape[1]):
        vl_Lab[0][n] = csc.RGB_to_Lab(pcs[0][n])

    ## Weight Setting
    weight = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 7, 5],
                       [3, 5, 7, 5, 3],
                       [1, 3, 5, 3, 1]])/48
    
    img = np.array(img, dtype=np.float32)
    x_RGB = np.copy(img)
    x_Lab = np.copy(img)
    v_RGB = np.copy(img)
    dist = np.zeros((1, pcs.shape[1]), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x_Lab[i][j] = csc.RGB_to_Lab(x_RGB[i][j])
            for n in range(pcs.shape[1]):
                dist[0][n] = distance(x_Lab[i][j], vl_Lab[0][n])
            min_n = np.argmin(dist[0])
            v_RGB[i][j] = pcs[0][min_n]
            error = x_RGB[i][j] - v_RGB[i][j]

            ## Error Diffuse
            if j < img.shape[1] - 1:
                x_RGB[i][j+1] += error*weight[2][3]
                if j < img.shape[1] - 2:
                    x_RGB[i][j+2] += error*weight[2][4]
            if i < img.shape[0] - 1:
                x_RGB[i+1][j] += error*weight[3][2]
                if j > 0:
                    x_RGB[i+1][j-1] += error*weight[3][1]
                    if j > 1:
                        x_RGB[i+1][j-2] += error*weight[3][0]
                if j < img.shape[1] - 1:
                    x_RGB[i+1][j+1] += error*weight[3][3]
                    if j < img.shape[1] - 2:
                        x_RGB[i+1][j+2] += error*weight[3][4]
                if i < img.shape[0] - 2:
                    x_RGB[i+2][j] += error*weight[4][2]
                    if j > 0:
                        x_RGB[i+2][j-1] += error*weight[4][1]
                        if j > 1:
                            x_RGB[i+2][j-2] += error*weight[4][0]
                    if j < img.shape[1] - 1:
                        x_RGB[i+2][j+1] += error*weight[4][3]
                        if j < img.shape[1] - 2:
                            x_RGB[i+2][j+2] += error*weight[4][4]
    
    return Image.fromarray(np.uint8(v_RGB))

### Direct Binary Search ###
def direct_binary_search(f0, gm, imax):
    # Initiation
    im = np.array(f0, dtype=np.float32)/255
    gm = np.array(gm, dtype=np.float32)/255
    rows, cols = im.shape

    fs = 7  # Gaussian filter size
    d = (fs - 1) / 6
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))
    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)

    CPP = np.zeros((13, 13))
    HalfCPPSize = 6
    CPP += convolve2d(GF, GF, mode='full')  # AutoCorrelation of Gaussian filter

    Err = gm - im
    CEP = correlate2d(Err, CPP, mode='full')  # Cross Correlation between Error and Gaussian

    EPS = 0
    iteration = 0
    while True:
        iteration += 1
        CountB = 0
        a0 = 0
        a1 = 0

        for i in range(rows):
            for j in range(cols):
                a0c = 0
                a1c = 0
                Cpx = 0
                Cpy = 0
                EPS_MIN = 0

                for y in range(-1, 2):
                    if i+y < 0 or i+y >= rows:
                        continue

                    for x in range(-1, 2):
                        if j+x < 0 or j+x >= cols:
                            continue

                        if y == 0 and x == 0:
                            if np.isclose(gm[i, j], 1.0):
                                a0 = -1
                                a1 = 0
                            else:
                                a0 = 1
                                a1 = 0
                        else:
                            if gm[i+y, j+x] != gm[i, j]:
                                if np.isclose(gm[i, j], 1.0):
                                    a0 = -1
                                    a1 = -a0
                                else:
                                    a0 = 1
                                    a1 = -a0
                            else:
                                a0 = 0
                                a1 = 0

                        EPS = (a0*a0 + a1*a1)*CPP[HalfCPPSize, HalfCPPSize] + 2*a0*a1*CPP[HalfCPPSize+y, HalfCPPSize+x]\
                              + 2*a0*CEP[i+HalfCPPSize, j+HalfCPPSize] + 2*a1*CEP[i+y+HalfCPPSize, j+x+HalfCPPSize]

                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y

                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize+1):
                        for x in range(-HalfCPPSize, HalfCPPSize+1):
                            CEP[i+y+HalfCPPSize, j+x+HalfCPPSize] += a0c * CPP[y+HalfCPPSize, x+HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize+1):
                        for x in range(-HalfCPPSize, HalfCPPSize+1):
                            CEP[i+y+Cpy+HalfCPPSize, j+x+Cpx+HalfCPPSize] += a1c * CPP[y+HalfCPPSize, x+HalfCPPSize]
                    gm[i, j] += a0c
                    gm[i+Cpy, j+Cpx] += a1c
                    CountB += 1
        if CountB == 0 or iteration >= imax:
            break

    gm *= 255
    # print(iteration)
    return Image.fromarray(np.uint8(gm)), CountB

### Color Direct Binary Search ###
def color_direct_binary_search(f0, gm, pcs, imax):    
    ## Initiation
    fm = np.array(f0, dtype=np.float32)
    gm = np.array(gm, dtype=np.float32)
    rows, cols, chnls = fm.shape
    
    ## Gaussian filter size
    fs = 7
    d = (fs - 1) / 6
    gaulen = int((fs - 1) / 2)
    Gf = np.zeros((fs, fs))
    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            Gf[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)

    ## AutoCorrelation of Gaussian filter
    Cpp = np.zeros((13, 13))
    hCpp = 6
    Cpp += convolve2d(Gf, Gf, mode='full')

    ## Cross Correlation between Error and Gaussian
    Err = gm - fm
    Cpe = np.zeros((rows + hCpp*2, cols + hCpp*2, chnls))
    for chnl in range(chnls):
        Cpe[:, :, chnl] = correlate2d(Err[:, :, chnl], Cpp, mode='full')

    iteration = 0
    while True:
        iteration += 1
        CountB = 0

        for i in range(rows):
            for j in range(cols):
                amLc, amac, ambc = 0, 0, 0
                anLc, anac, anbc = 0, 0, 0
                Cpx, Cpy = 0, 0
                Delta_Emin = 0

                for k in range(pcs.shape[1]):
                    amL = pcs[0, k, 0] - gm[i, j, 0]
                    ama = pcs[0, k, 1] - gm[i, j, 1]
                    amb = pcs[0, k, 2] - gm[i, j, 2]
                    anL, ana, anb = 0, 0, 0

                    Delta_E = amL*amL*Cpp[hCpp, hCpp] + 2*amL*Cpe[i+hCpp, j+hCpp, 0]\
                            + ama*ama*Cpp[hCpp, hCpp] + 2*ama*Cpe[i+hCpp, j+hCpp, 1]\
                            + amb*amb*Cpp[hCpp, hCpp] + 2*amb*Cpe[i+hCpp, j+hCpp, 2]

                    if Delta_Emin > Delta_E:
                        Delta_Emin = Delta_E
                        amLc, amac, ambc = amL, ama, amb
                        anLc, anac, anbc = anL, ana, anb

                for x in range(-1, 2):
                    if i+x < 0 or i+x >= rows:
                        continue
                    for y in range(-1, 2):
                        if j+y < 0 or j+y >= cols:
                            continue
                        
                        if x == 0 and y == 0:
                            continue

                        else:
                            if gm[i+x, j+y, 0] != gm[i, j, 0] and gm[i+x, j+y, 1] != gm[i, j, 1] and gm[i+x, j+y, 2] != gm[i, j, 2]:
                                amL = gm[i+x, j+y, 0] - gm[i, j, 0]
                                ama = gm[i+x, j+y, 1] - gm[i, j, 1]
                                amb = gm[i+x, j+y, 2] - gm[i, j, 2]
                                anL, ana, anb = -amL, -ama, -amb 
                            else:
                                continue
                
                        Delta_E = (amL*amL + anL*anL)*Cpp[hCpp, hCpp] + 2*amL*anL*Cpp[hCpp+x, hCpp+y] + 2*amL*Cpe[i+hCpp, j+hCpp, 0] + 2*anL*Cpe[i+x+hCpp, j+y+hCpp, 0]\
                                + (ama*ama + ana*ana)*Cpp[hCpp, hCpp] + 2*ama*ana*Cpp[hCpp+x, hCpp+y] + 2*ama*Cpe[i+hCpp, j+hCpp, 1] + 2*ana*Cpe[i+x+hCpp, j+y+hCpp, 1]\
                                + (amb*amb + anb*anb)*Cpp[hCpp, hCpp] + 2*amb*anb*Cpp[hCpp+x, hCpp+y] + 2*amb*Cpe[i+hCpp, j+hCpp, 2] + 2*anb*Cpe[i+x+hCpp, j+y+hCpp, 2]
                    
                        if Delta_Emin > Delta_E:
                            Delta_Emin = Delta_E
                            amLc, amac, ambc = amL, ama, amb
                            anLc, anac, anbc = anL, ana, anb
                            Cpx, Cpy = x, y
                            
                if Delta_Emin < 0:
                    for x in range(-hCpp, hCpp+1):
                        for y in range(-hCpp, hCpp+1):
                            Cpe[i+x+hCpp, j+y+hCpp, 0] += amLc*Cpp[x+hCpp, y+hCpp]
                            Cpe[i+x+hCpp, j+y+hCpp, 1] += amac*Cpp[x+hCpp, y+hCpp]
                            Cpe[i+x+hCpp, j+y+hCpp, 2] += ambc*Cpp[x+hCpp, y+hCpp]
                    gm[i, j, 0] += amLc
                    gm[i, j, 1] += amac
                    gm[i, j, 2] += ambc
                    if anLc != 0 or anac != 0 or anbc != 0:
                        for x in range(-hCpp, hCpp+1):
                            for y in range(-hCpp, hCpp+1):
                                Cpe[i+x+Cpx+hCpp, j+y+Cpy+hCpp, 0] += anLc*Cpp[x+hCpp, y+hCpp]
                                Cpe[i+x+Cpx+hCpp, j+y+Cpy+hCpp, 1] += anac*Cpp[x+hCpp, y+hCpp]
                                Cpe[i+x+Cpx+hCpp, j+y+Cpy+hCpp, 2] += anbc*Cpp[x+hCpp, y+hCpp]
                        gm[i+Cpx, j+Cpy, 0] += anLc
                        gm[i+Cpx, j+Cpy, 1] += anac
                        gm[i+Cpx, j+Cpy, 2] += anbc
                    CountB += 1
        
        if CountB == 0 or iteration >= imax:
            break

    # print(iteration)
    gm = np.clip(gm, 0, 255)
    return Image.fromarray(np.uint8(gm)), CountB

### Random Tiled Block Color Direct Binary Search ###
def Sequence_List(sigma):
    out = []
    img = Image.open(f"Input/Mask/{sigma}.tiff")
    img = np.array(img)
    # img = np.array([
    #     [ 0, 32,  8, 40,  2, 34, 10, 42],
    #     [48, 16, 56, 24, 50, 18, 58, 26],
    #     [12, 44,  4, 36, 14, 46,  6, 38],
    #     [60, 28, 52, 20, 62, 30, 54, 22],
    #     [ 3, 35, 11, 43,  1, 33,  9, 41],
    #     [51, 19, 59, 27, 49, 17, 57, 25],
    #     [15, 47,  7, 39, 13, 45,  5, 37],
    #     [63, 31, 55, 23, 61, 29, 53, 21]])

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         out.append(img[i, j])
    ## Z's Method
    count = 0
    while count <= 255:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] == count:
                    out.append(i * img.shape[1] + j)
                    count += 1
    return out

def ToggleC(gm, pcs, Cpp, Cpe, hCpp, i, j, convergence):
    amLc, amac, ambc = 0, 0, 0
    Delta_Emin = 0
    for cl in range(pcs.shape[1]):
        amL = pcs[0, cl, 0] - gm[i, j, 0]
        ama = pcs[0, cl, 1] - gm[i, j, 1]
        amb = pcs[0, cl, 2] - gm[i, j, 2]

        Delta_E = amL*amL*Cpp[hCpp, hCpp] + 2*amL*Cpe[i+hCpp, j+hCpp, 0]\
                + ama*ama*Cpp[hCpp, hCpp] + 2*ama*Cpe[i+hCpp, j+hCpp, 1]\
                + amb*amb*Cpp[hCpp, hCpp] + 2*amb*Cpe[i+hCpp, j+hCpp, 2]

        if Delta_Emin > Delta_E:
            Delta_Emin = Delta_E
            amLc, amac, ambc = amL, ama, amb
                
    if Delta_Emin < 0:
        for x in range(-hCpp, hCpp+1):
            for y in range(-hCpp, hCpp+1):
                Cpe[i+x+hCpp, j+y+hCpp, 0] += amLc*Cpp[x+hCpp, y+hCpp]
                Cpe[i+x+hCpp, j+y+hCpp, 1] += amac*Cpp[x+hCpp, y+hCpp]
                Cpe[i+x+hCpp, j+y+hCpp, 2] += ambc*Cpp[x+hCpp, y+hCpp]
        gm[i, j, 0] += amLc
        gm[i, j, 1] += amac
        gm[i, j, 2] += ambc
        convergence += 1
    return gm, Cpe, convergence

def RTBCDBS_OnlyToggle(fm, gm, pcs, sigma, size, imax):
    fm = np.array(fm, dtype=np.float32)
    gm = np.array(gm, dtype=np.float32)
    rows, cols, chls = fm.shape

    sig = 18
    seq = Sequence_List(sig)

    gs = (size + 1)//2
    length = (gs - 1)//2
    Gf = np.zeros((gs, gs))
    for x in range(-length, length + 1):
        for y in range(-length, length + 1):
            c = (m.pow(x, 2) + m.pow(y, 2))/(2*m.pow(sigma, 2))
            Gf[x + length, y + length] = m.exp(-c)/(2*m.pi*m.pow(sigma, 2))
    hCpp = (size - 1)//2
    Cpp = convolve2d(Gf, Gf, mode='full')

    Err = gm - fm
    Cpe = np.zeros((rows + hCpp*2, cols + hCpp*2, chls))
    for chnl in range(chls):
        Cpe[:, :, chnl] = correlate2d(Err[:, :, chnl], Cpp, mode='full')

    iteration = 0
    while True:
        iteration += 1
        convergence = 0

        for i in range(0, rows - rows%16, 16):
            for j in range(0, cols - cols%16, 16):
                begin = np.random.randint(0, 255)
                for s in range(begin, 256):
                    k = seq[s]//16
                    l = seq[s]%16
                    gm, Cpe, convergence = ToggleC(gm, pcs, Cpp, Cpe, hCpp, i+k, j+l, convergence)
                for s in range(begin):
                    k = seq[s]//16
                    l = seq[s]%16
                    gm, Cpe, convergence = ToggleC(gm, pcs, Cpp, Cpe, hCpp, i+k, j+l, convergence)
        for i in range(rows):
            if i < (rows - rows%16):
                for j in range(cols - cols%16, cols):
                    gm, Cpe, convergence = ToggleC(gm, pcs, Cpp, Cpe, hCpp, i, j, convergence)
            else:
                for j in range(cols):
                    gm, Cpe, convergence = ToggleC(gm, pcs, Cpp, Cpe, hCpp, i, j, convergence)

        if convergence == 0 or iteration >= imax:
            break

    gm = np.clip(gm, 0, 255)
    return Image.fromarray(np.uint8(gm)), convergence

### Random Tiled Block Direct Binary Search ###
def ToggleB(gm, Cpp, Cpe, hCpp, i, j, convergence):
    amc = 0
    Delta_Emin = 0
    if np.isclose(gm[i, j], 1.0):
        am = -1
    else:
        am = 1

    Delta_E = am*am*Cpp[hCpp, hCpp] + 2*am*Cpe[i+hCpp, j+hCpp]

    if Delta_Emin > Delta_E:
        Delta_Emin = Delta_E
        amc = am
                
    if Delta_Emin < 0:
        for x in range(-hCpp, hCpp+1):
            for y in range(-hCpp, hCpp+1):
                Cpe[i+x+hCpp, j+y+hCpp] += amc*Cpp[x+hCpp, y+hCpp]

        gm[i, j] += amc
        convergence += 1
    return gm, Cpe, convergence

def RTBDBS(fm, gm, sigma, size, imax):
    fm = np.array(fm, dtype=np.float32)/255
    gm = np.array(gm, dtype=np.float32)/255
    rows, cols = fm.shape

    sig = 18
    seq = Sequence_List(sig)

    gs = (size + 1)//2
    length = (gs - 1)//2
    Gf = np.zeros((gs, gs))
    for x in range(-length, length + 1):
        for y in range(-length, length + 1):
            c = (m.pow(x, 2) + m.pow(y, 2))/(2*m.pow(sigma, 2))
            Gf[x + length, y + length] = m.exp(-c)/(2*m.pi*m.pow(sigma, 2))
    hCpp = (size - 1)//2
    Cpp = convolve2d(Gf, Gf, mode='full')

    Err = gm - fm
    Cpe = correlate2d(Err, Cpp, mode='full')

    iteration = 0
    while True:
        iteration += 1
        convergence = 0

        for i in range(0, rows - rows%16, 16):
            for j in range(0, cols - cols%16, 16):
                begin = np.random.randint(0, 255)
                for s in range(begin, 256):
                    k = seq[s]//16
                    l = seq[s]%16
                    gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i+k, j+l, convergence)
                for s in range(begin):
                    k = seq[s]//16
                    l = seq[s]%16
                    gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i+k, j+l, convergence)
        for i in range(rows):
            if i < (rows - rows%16):
                for j in range(cols - cols%16, cols):
                    gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i, j, convergence)
            else:
                for j in range(cols):
                    gm, Cpe, convergence = ToggleB(gm, Cpp, Cpe, hCpp, i, j, convergence)

        # print(iteration)

        if convergence == 0 or iteration >= imax:
            break

    gm = np.clip(gm*255, 0, 255)
    return Image.fromarray(np.uint8(gm)), convergence

def f_ini(sigma):
    filter = np.zeros((5, 5), dtype=np.float32)
    rows, cols = filter.shape
    for j in range(rows):
        for i in range(cols):
            filter[j, i] = np.exp(-(np.power(i - cols / 2, 2) + np.power(j - rows / 2, 2)) / (2.0 * np.power(sigma, 2.0)))
    return filter

def convolution_p(img, kernal):
    rows, cols = img.shape[0], img.shape[1]
    out = np.zeros((rows, cols), dtype=np.float32)
    for j in range(2, rows - 2):
        for i in range(2, cols - 2):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    out[j, i] += img[j + y, i + x] * kernal[y + 2, x + 2]
    return out

def do_3_times(img, out, imax, f_kernal, seed):
    img = np.array(img, dtype=np.uint8)
    out = np.array(out, dtype=np.uint8)
    fil_img = convolution_p(img, f_kernal)
    fil_out = convolution_p(out, f_kernal)
    error = fil_out - fil_img
    f_kernal *= 255
    for t in range(imax):
        cpp_0 = f_kernal[2, 2]
        for j in range(2, img.shape[0] - 15, 16):
            for i in range(2, img.shape[1] - 15, 16):
                begin = np.random.randint(0, 255)
                for m in range(begin, 256):
                    k = seed[m] % 16
                    l = seed[m] // 16
                    if out[j + l, i + k] == 0:
                        if error[j + l, i + k] < -0.5 * cpp_0:
                            for y in range(-2, 3):
                                for x in range(-2, 3):
                                    error[j + l + y, i + k + x] += f_kernal[y + 2, x + 2]
                            out[j + l, i + k] = 255
                    elif out[j + l, i + k] == 255:
                        if error[j + l, i + k] > 0.5 * cpp_0:
                            for y in range(-2, 3):
                                for x in range(-2, 3):
                                    error[j + l + y, i + k + x] -= f_kernal[y + 2, x + 2]
                            out[j + l, i + k] = 0
                for m in range(begin):
                    k = seed[m] % 16
                    l = seed[m] // 16
                    if out[j + l, i + k] == 0:
                        if error[j + l, i + k] < -0.5 * cpp_0:
                            for y in range(-2, 3):
                                for x in range(-2, 3):
                                    error[j + l + y, i + k + x] += f_kernal[y + 2, x + 2]
                            out[j + l, i + k] = 255
                    elif out[j + l, i + k] == 255:
                        if error[j + l, i + k] > 0.5 * cpp_0:
                            for y in range(-2, 3):
                                for x in range(-2, 3):
                                    error[j + l + y, i + k + x] -= f_kernal[y + 2, x + 2]
                            out[j + l, i + k] = 0
    return Image.fromarray(np.uint8(out))

def block_DBS_homo_random_p(img, out, sigma, imax):
    rr, gg, bb = img.split()
    r, g, b = out.split()
    f_kernal = f_ini(sigma)
    seed = Sequence_List(18)
    r = do_3_times(rr, r, imax, f_kernal, seed)
    g = do_3_times(gg, g, imax, f_kernal, seed)
    b = do_3_times(bb, b, imax, f_kernal, seed)
    out = Image.merge("RGB", (r, g, b))
    return out
