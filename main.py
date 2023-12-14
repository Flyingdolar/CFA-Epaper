import math as M
import numpy as np
import Resize as rs
import CCM_Process as ccm_p
import CFA_Pattern as cfa_p
import To_Grayscale as tg

def main():

    # First step: Resize image to 1872*1404
    rs_input_path = "input/img161.jpg"
    rs_output_path = "output/img161_2.bmp"
    target_width = 1872
    target_height = 1404

    rs.resize_image(rs_input_path, rs_output_path, target_width, target_height)

    # Second step: CCM process (with step 3 and step 4 / without step 3 and step 4)
    ccm_input_path = rs_output_path
    ccm_output_path = "output/img161_ccm.bmp"
    ccm_p.apply_color_correction(ccm_input_path, ccm_output_path)

    # Third step: HDR (only lab) (with or without)
    HDR_input_path = ccm_output_path
    # NOT DONE YET

    # Fourth step: CFA pattern
    CFA_input_path = HDR_input_path
    CFA_output_path = "output/img161_cfa.bmp"
    cfa_p.apply_CFA(CFA_input_path, CFA_output_path)

    # Final step: Convert to grayscale
    gray_input_path = CFA_output_path
    gray_output_path = "output/img161_gray.bmp"
    tg.convert_to_grayscale(gray_input_path, gray_output_path)


if __name__ == "__main__":
    main()
