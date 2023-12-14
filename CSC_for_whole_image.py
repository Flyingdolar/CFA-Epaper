import math as M
import numpy as np


def RGB_to_XYZ_image(input_image, output_image):
    R = input_image[:, :, 0]
    G = input_image[:, :, 1]
    B = input_image[:, :, 2]
    X = 0.412453 * (R / 255) + 0.357580 * (G / 255) + 0.180423 * (B / 255)
    Y = 0.212671 * (R / 255) + 0.715160 * (G / 255) + 0.072169 * (B / 255)
    Z = 0.019334 * (R / 255) + 0.119193 * (G / 255) + 0.950227 * (B / 255)
    output_image[:, :, 0] = X
    output_image[:, :, 1] = Y
    output_image[:, :, 2] = Z
    return output_image


def XYZ_to_RGB_image(input_image, output_image):
    X = input_image[:, :, 0]
    Y = input_image[:, :, 1]
    Z = input_image[:, :, 2]
    R = (3.240479 * X - 1.537150 * Y - 0.498535 * Z) * 255
    G = (-0.969256 * X + 1.875992 * Y + 0.041556 * Z) * 255
    B = (0.055648 * X - 0.204043 * Y + 1.057311 * Z) * 255
    output_image[:, :, 0] = R
    output_image[:, :, 1] = G
    output_image[:, :, 2] = B
    return output_image


def XYZ_to_Lab_image(input_image, output_image):
    X = input_image[:, :, 0]
    Y = input_image[:, :, 1]
    Z = input_image[:, :, 2]
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
    output_image[:, :, 0] = L
    output_image[:, :, 1] = a
    output_image[:, :, 2] = b
    return output_image


def Lab_to_XYZ_image(input_image, output_image):
    L = input_image[:, :, 0]
    a = input_image[:, :, 1]
    b = input_image[:, :, 2]
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
    output_image[:, :, 0] = X
    output_image[:, :, 1] = Y
    output_image[:, :, 2] = Z
    return output_image

    
def XYZ_to_Yxy_image(input_image, output_image):
    X = input_image[:, :, 0]
    Y = input_image[:, :, 1]
    Z = input_image[:, :, 2]
    if X + Y + Z == 0:
        output_image[:, :, 0] = 0
        output_image[:, :, 1] = 0.3127
        output_image[:, :, 2] = 0.3290
    else:
        output_image[:, :, 0] = Y
        output_image[:, :, 1] = X / (X + Y + Z)
        output_image[:, :, 2] = Y / (X + Y + Z)
    return output_image

    
def Yxy_to_XYZ_image(input_image, output_image):
    ipnut_image = ipnut_image.astype(np.float32)
    Y = ipnut_image[:, :, 0]
    x = ipnut_image[:, :, 1]
    y = ipnut_image[:, :, 2]
    if y == 0:
        output_image[:, :, 0] = 0
        output_image[:, :, 1] = 0
        output_image[:, :, 2] = 0
    else:
        output_image[:, :, 0] = (x / y) * Y
        output_image[:, :, 1] = Y
        output_image[:, :, 2] = ((1 - x - y) / y) * Y
    return output_image

def XYZ_to_Luv_image(input_image, output_image):
    X = input_image[:, :, 0]
    Y = input_image[:, :, 1]
    Z = input_image[:, :, 2]
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
    output_image[:, :, 0] = L
    output_image[:, :, 1] = u
    output_image[:, :, 2] = v
    return output_image


def XYZ_to_YyCxCz_image(input_image, output_image):
    X = input_image[:, :, 0]
    Y = input_image[:, :, 1]
    Z = input_image[:, :, 2]
    Xn = 0.950456
    Yn = 1.0
    Zn = 1.088754
    Yy = 116 * (Y / Yn) - 16
    Cx = 500.0 * (X / Xn - Y / Yn)
    Cz = 200.0 * (Y / Yn - Z / Zn)
    output_image[:, :, 0] = Yy
    output_image[:, :, 1] = Cx
    output_image[:, :, 2] = Cz
    return output_image


def YyCxCz_to_XYZ_image(input_image, output_image):
    Yy = input_image[:, :, 0]
    Cx = input_image[:, :, 1]
    Cz = input_image[:, :, 2]
    Xn = 0.950456
    Yn = 1
    Zn = 1.088754
    Y = ((Yy + 16) / 116) * Yn
    X = (Cx / 500 + Y / Yn) * Xn
    Z = -(Cz / 200 - Y / Yn) * Zn
    output_image[:, :, 0] = X
    output_image[:, :, 1] = Y
    output_image[:, :, 2] = Z
    return output_image


