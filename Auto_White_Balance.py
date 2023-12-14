import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

x_data = np.array([
    [255, 255, 255],  # White
    [243, 243, 242],  # No.19
    [200, 200, 200],  # No.20
    [160, 160, 160],  # No.21
    [122, 122, 121],  # No.22
    [ 85,  85,  85],  # No.23
    [ 52,  52,  52],  # No.24
    [  0,   0,   0]])  # Black

y_data = np.array([
    [39.69083023,	46.14099121,	44.42738724],   # White
    [35.72702789,	42.24011993,	41.31967926],   # No.19
    [28.87051964,	35.17556381,	35.45739746],   # No.20
    [22.02905846,	26.96634865,	27.63151169],   # No.21
    [15.35199261,	18.85037422,	19.68402290],   # No.22
    [10.66071129,	13.15429115,	14.13704681],   # No.23
    [7.933727741,	9.694092751,	10.58777714],   # No.24
    [3.949607134,	4.656770706,	5.483669281]])   # Black

slopeR, interceptR, r_value, p_value, std_err = linregress(x_data[:, 0], y_data[:, 0])
slopeG, interceptG, r_value, p_value, std_err = linregress(x_data[:, 1], y_data[:, 1])
slopeB, interceptB, r_value, p_value, std_err = linregress(x_data[:, 2], y_data[:, 2])

print(f"Slope R: {slopeR}, Intercept R: {interceptR}")
print(f"Slope G: {slopeG}, Intercept G: {interceptG}")
print(f"Slope B: {slopeB}, Intercept B: {interceptB}")

y_predR = slopeR * x_data[:, 0] + interceptR
y_predG = slopeG * x_data[:, 1] + interceptG
y_predB = slopeB * x_data[:, 2] + interceptB

plt.scatter(x_data[:, 0], y_data[:, 0], label='R', color='#af363c')
plt.scatter(x_data[:, 1], y_data[:, 1], label='G', color='#469449')
plt.scatter(x_data[:, 2], y_data[:, 2], label='B', color='#383d96')

plt.plot(x_data[:, 0], y_predR, color='#af363c', label='Regression Line')
plt.plot(x_data[:, 1], y_predG, color='#469449', label='Regression Line')
plt.plot(x_data[:, 2], y_predB, color='#383d96', label='Regression Line')

plt.xlabel('Color Checker')
plt.ylabel('CFA Epaper')
plt.title('Linear Regression')
plt.legend()
plt.show()
