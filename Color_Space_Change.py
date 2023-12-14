import math as M
import numpy as np


def RGB_to_XYZ(R, G, B):
    X = 0.412453 * (R / 255) + 0.357580 * (G / 255) + 0.180423 * (B / 255)
    Y = 0.212671 * (R / 255) + 0.715160 * (G / 255) + 0.072169 * (B / 255)
    Z = 0.019334 * (R / 255) + 0.119193 * (G / 255) + 0.950227 * (B / 255)
    return X, Y, Z


def XYZ_to_RGB(X, Y, Z):
    R = (3.240479 * X - 1.537150 * Y - 0.498535 * Z) * 255
    G = (-0.969256 * X + 1.875992 * Y + 0.041556 * Z) * 255
    B = (0.055648 * X - 0.204043 * Y + 1.057311 * Z) * 255
    return R, G, B


def XYZ_to_Lab(X, Y, Z):
    Xn = 0.950456
    Yn = 1
    Zn = 1.088754
    fx = (
        M.pow(X / Xn, 1 / 3)
        if (X / Xn) > M.pow(6 / 29, 3)
        else (1 / 3) * M.pow(29 / 6, 2) * (X / Xn) + 16 / 116
    )
    fy = (
        M.pow(Y / Yn, 1 / 3)
        if (Y / Yn) > M.pow(6 / 29, 3)
        else (1 / 3) * M.pow(29 / 6, 2) * (Y / Yn) + 16 / 116
    )
    fz = (
        M.pow(Z / Zn, 1 / 3)
        if (Z / Zn) > M.pow(6 / 29, 3)
        else (1 / 3) * M.pow(29 / 6, 2) * (Z / Zn) + 16 / 116
    )
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


def Lab_to_XYZ(L, a, b):
    Xn = 0.950456
    Yn = 1
    Zn = 1.088754
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    X = (
        Xn * M.pow(fx, 3)
        if fx > (6 / 29)
        else (fx - 16 / 116) * 3 * M.pow(6 / 29, 2) * Xn
    )
    Y = (
        Yn * M.pow(fy, 3)
        if fy > (6 / 29)
        else (fy - 16 / 116) * 3 * M.pow(6 / 29, 2) * Yn
    )
    Z = (
        Zn * M.pow(fz, 3)
        if fz > (6 / 29)
        else (fz - 16 / 116) * 3 * M.pow(6 / 29, 2) * Zn
    )
    return X, Y, Z


def XYZ_to_Yxy(X, Y, Z, D65x=0.3127, D65y=0.3290):
    if X + Y + Z == 0:
        return 0, D65x, D65y
    else:
        return Y, X / (X + Y + Z), Y / (X + Y + Z)


def Yxy_to_XYZ(Y, x, y):
    if y == 0:
        return 0, 0, 0
    else:
        X = (x / y) * Y
        Z = ((1 - x - y) / y) * Y
        return X, Y, Z


def XYZ_to_Luv(X, Y, Z):
    Xn = 0.950456
    Yn = 1
    Zn = 1.088754
    un_ = 0.2009
    vn_ = 0.4610
    u_ = 4 * X / (X + 15 * Y + 3 * Z)
    v_ = 9 * X / (X + 15 * Y + 3 * Z)
    L = (
        M.pow((29 / 3), 3) * (Y / Yn)
        if (Y / Yn) <= M.pow(6 / 29, 3)
        else 116 * M.pow(Y / Yn, 1 / 3) - 16
    )
    u = 13 * L * (u_ - un_)
    v = 13 * L * (v_ - vn_)
    return L, u, v


def XYZ_to_YyCxCz(X, Y, Z):
    Xn = 0.950456
    Yn = 1.0
    Zn = 1.088754
    Yy = 116 * (Y / Yn) - 16
    Cx = 500.0 * (X / Xn - Y / Yn)
    Cz = 200.0 * (Y / Yn - Z / Zn)
    return Yy, Cx, Cz


def YyCxCz_to_XYZ(Yy, Cx, Cz):
    Xn = 0.950456
    Yn = 1
    Zn = 1.088754
    Y = ((Yy + 16) / 116) * Yn
    X = (Cx / 500 + Y / Yn) * Xn
    Z = -(Cz / 200 - Y / Yn) * Zn
    return X, Y, Z


## Linear sRGB, Reference: https://en.wikipedia.org/wiki/SRGB
def sRGB_to_linearRGB(input):
    ThrLin2Gamma = 0.04045
    output = np.copy(input)
    for chnl in range(3):
        temp = input[chnl] / 255
        if temp > ThrLin2Gamma:
            output[chnl] = M.pow((temp + 0.055) / 1.055, 2.4) * 255
        else:
            output[chnl] = (temp / 12.92) * 255
    return output


def linearRGB_to_sRGB(input):
    ThrLin2Gamma = 0.0031308
    output = np.copy(input)
    for channel in range(3):
        temp = output[channel] / 255
        if temp > ThrLin2Gamma:
            output[channel] = (1.055 * M.pow(temp, 1 / 2.4) - 0.055) * 255
        else:
            output[channel] = 12.92 * temp * 255
    return output


## OkLab, Reference: https://bottosson.github.io/posts/oklab/#converting-from-xyz-to-oklab
def sRGB_to_OkLab(input):
    # output = np.copy(input)
    input = sRGB_to_linearRGB(input)
    lR, lG, lB = input[0], input[1], input[2]
    X, Y, Z = RGB_to_XYZ(lR, lG, lB)
    l = 0.8189330101 * X + 0.3618667424 * Y - 0.1288597137 * Z
    m = 0.0329845436 * X + 0.9293118715 * Y + 0.0361456387 * Z
    s = 0.0482003018 * X + 0.2643662691 * Y + 0.6338517070 * Z
    l_, m_, s_ = M.pow(l, 1 / 3), M.pow(m, 1 / 3), M.pow(s, 1 / 3)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return [L, a, b]


def OkLab_to_sRGB(input):
    L, a, b = input[0], input[1], input[2]
    l_ = 1 * L + 0.39633779 * a + 0.21580376 * b
    m_ = 1.00000001 * L - 0.10556134 * a - 0.06385417 * b
    s_ = 1.00000005 * L - 0.08948418 * a - 1.29148554 * b
    l, m, s = M.pow(l_, 3), M.pow(m_, 3), M.pow(s_, 3)
    X = 1.22701385 * l - 0.55779998 * m + 0.28125615 * s
    Y = -0.04058018 * l + 1.11225687 * m - 0.07167668 * s
    Z = -0.07638128 * l - 0.42148198 * m + 1.58616322 * s
    lR, lG, lB = XYZ_to_RGB(X, Y, Z)
    input = linearRGB_to_sRGB([lR, lG, lB])
    return input


## Mixed Function
def sRGB_to_Lab(input):
    output = np.copy(input)
    lR, lG, lB = sRGB_to_linearRGB(input)
    X, Y, Z = RGB_to_XYZ(lR, lG, lB)
    L, a, b = XYZ_to_Lab(X, Y, Z)
    output = [L / 100, (a + 128) / 255, (b + 128) / 255]
    return output


def Lab_to_sRGB(input):
    output = np.copy(input)
    X, Y, Z = Lab_to_XYZ(input[0] * 100, input[1] * 255 - 128, input[2] * 255 - 128)
    lR, lG, lB = XYZ_to_RGB(X, Y, Z)
    output = linearRGB_to_sRGB([lR, lG, lB])
    return output


def RGB_to_Lab(input):
    output = np.copy(input)
    # lR, lG, lB = sRGB_to_linearRGB(input)
    X, Y, Z = RGB_to_XYZ(input[0], input[1], input[2])
    L, a, b = XYZ_to_Lab(X, Y, Z)
    output = [L / 100, (a + 128) / 255, (b + 128) / 255]
    return output


def sRGB_to_Luv(input):
    output = np.copy(input)
    lR, lG, lB = sRGB_to_linearRGB(input)
    X, Y, Z = RGB_to_XYZ(lR, lG, lB)
    L, u, v = XYZ_to_Luv(X, Y, Z)
    output = [L, u, v]
    return output


def RGB_to_Luv(input):
    output = np.copy(input)
    X, Y, Z = RGB_to_XYZ(input[0], input[1], input[2])
    L, u, v = XYZ_to_Luv(X, Y, Z)
    output = [L, u, v]
    return output


def RGB_to_YyCxCz(input):
    output = np.copy(input)
    X, Y, Z = RGB_to_XYZ(input[0], input[1], input[2])
    Yy, Cx, Cz = XYZ_to_YyCxCz(X, Y, Z)
    output = Yy, Cx, Cz
    return output


def YyCxCz_to_RGB(input):
    output = np.copy(input)
    X, Y, Z = YyCxCz_to_XYZ(input[0], input[1], input[2])
    R, G, B = XYZ_to_RGB(X, Y, Z)
    output = R, G, B
    return output


# matrix1 = [
#     [0.8189330101, 0.3618667424, -0.1288597137],
#     [0.0329845436, 0.9293118715,  0.0361456387],
#     [0.0482003018, 0.2643662691,  0.6338517070]
# ]
# matrix2 = [
#     [0.2104542553,  0.7936177850, -0.0040720468],
#     [1.9779984951, -2.4285922050,  0.4505937099],
#     [0.0259040371,  0.7827717662, -0.8086757660]
# ]
# print(np.linalg.inv(matrix2))

# M1-1
# [[ 1.22701385 -0.55779998  0.28125615]
#  [-0.04058018  1.11225687 -0.07167668]
#  [-0.07638128 -0.42148198  1.58616322]]
# M2-1
# [[ 1.          0.39633779  0.21580376]
#  [ 1.00000001 -0.10556134 -0.06385417]
#  [ 1.00000005 -0.08948418 -1.29148554]]
