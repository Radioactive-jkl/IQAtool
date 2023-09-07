# std
import os
import argparse
from argparse import RawTextHelpFormatter
# 3p
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate(folder_o, folder_p):
    count = 0
    ssim_values = []
    psnr_values = []

    for filename_o in os.listdir(folder_o):
        if filename_o.endswith(('.jpg', '.jpeg', '.png')):
            filepath_o = os.path.join(folder_o, filename_o)

            # filename_p may be different from filename_o.  
            filename_p = filename_o
            filepath_p = os.path.join(folder_p, filename_p)

            if os.path.isfile(filepath_p):
                img_o = cv2.imread(filepath_o)
                img_p = cv2.imread(filepath_p)

                # Calculate PSNR
                #mse = np.mean((img_o - img_p) ** 2)
                #psnr_value = 10 * np.log10((255 ** 2) / mse)
                psnr_value = psnr(img_o, img_p)
                psnr_values.append(psnr_value)

                # Calculate SSIM
                #ssim_value = ssim(img_o, img_p)
                ssim_channels = []
                for channel in range(3):
                    channel_o = img_o[:, :, channel]
                    channel_p = img_p[:, :, channel]
                    ssim_channel = ssim(channel_o, channel_p)
                    ssim_channels.append(ssim_channel)

                ssim_value = np.mean(ssim_channels)
                ssim_values.append(ssim_value)

                count = count + 1
                print(count, '\t PSNR: ', psnr_value, '\t SSIM: ', ssim_value, '\t')

    
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    return mean_psnr, mean_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python code for PSNR, SSIM",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-o", '--origin', default='./origin', type=str,
                        help="folder path to original images or ground truth.")
    parser.add_argument("-p", '--proccessed', default='./proccessed', type=str,
                        help="folder path to proccessed images.")
    
    args = parser.parse_args()
    
    mean_psnr, mean_ssim = calculate(args.origin, args.proccessed)

    print('Average SSIM:', mean_ssim)
    print('Average PSNR:', mean_psnr)