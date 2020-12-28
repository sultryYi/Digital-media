import numpy as np
import cv2
import math


class LPG_PCA:
    def __init__(cls, fileName):
        cls.img = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        cls.K = 3
        cls.L = 7
        cls.m = cls.K * cls.K
        cls.n = 10
        cls.row = cls.img.shape[0]
        cls.column = cls.img.shape[1]

    def run(cls):
        # 对每个像素点，先做出L*L矩阵和K*K矩阵
        # K*K矩阵是每个样本x
        # 边界不管

        # 所观察的像素点坐标范围为:
        # [row_start, row_end)
        # [column_start, column_end)
        L_radius = int(cls.L / 2)
        K_radius = int(cls.K / 2)

        row_start = L_radius
        row_end = cls.row - row_start

        column_start = L_radius
        column_end = cls.column - column_start

        cls.denoise = cls.img.copy()

        for i in range(row_start, row_end):
            for j in range(column_start, column_end):
                focus = cls.img[i - L_radius:i + L_radius + 1, j - L_radius:j + L_radius + 1].copy()
                x0 = focus[L_radius - K_radius:L_radius - K_radius + 3, L_radius -
                           K_radius:L_radius - K_radius + 3].reshape((-1, 1))
                coordinate_e = {}
                X = x0.copy()
                temp = [[[] for one in range(cls.L - cls.K + 1)]
                        for two in range(cls.L - cls.K + 1)]

                # 计算ei，选出最小的那10个(n个)作为样本，保存在X中
                # X的shape为(m, n)
                for fcRow in range(cls.L - cls.K + 1):
                    for fcCol in range(cls.L - cls.K + 1):
                        if fcRow == fcCol and fcCol == L_radius:
                            continue
                        xi = focus[fcRow:fcRow + 3, fcCol:fcCol + 3].reshape((-1, 1))
                        ei = np.mean(x0 - np.power(xi, 2))
                        coordinate_e[(fcRow, fcCol)] = ei
                        temp[fcRow][fcCol] = xi

                sortItems = sorted(coordinate_e.items(), key=lambda kv: kv[1])
                for item in sortItems[0:cls.n - 1]:
                    X = np.hstack([X, temp[item[0][0]][item[0][1]]])
                # print(X.shape)

                # 均值归一化
                average = np.mean(X, axis=1).reshape((-1, 1))
                X = X - average

                # PCA
                covMat = np.cov(X)
                vals, vects = np.linalg.eig(np.mat(covMat))
                valIndice = np.argsort(vals)
                n_valIndice = valIndice[-1:-(cls.n + 1):-1]
                transform = vects[:, n_valIndice]
                Y = np.dot(transform, X)

                # transforming back
                inverse_transform = np.linalg.inv(transform)
                X_back = np.dot(inverse_transform, Y)
                X_ave = X_back + average
                pixel = X_ave[int(cls.m / 2), 0]
                cls.denoise[i][j] = int(pixel)

    def getResult(cls):
        return cls.denoise.copy()


class Check:
    def __init__(cls, resource, target):
        cls.resource = resource
        cls.target = target

        cls.MSE = np.mean(np.power(resource - target, 2))

    def getPSNR(cls):
        if cls.MSE == 0:
            cls.MSE = 0.01
        psnr = 10 * math.log10((2**8 - 1)**2 / cls.MSE)
        return psnr

    def getSSIM(cls):
        L = 256
        k1 = 0.01
        k2 = 0.03

        c1 = (k1 * L)**2
        c2 = (k2 * L)**2

        meanX = np.mean(cls.resource)
        meanY = np.mean(cls.target)

        sigmaX = np.var(cls.resource)
        sigmaY = np.var(cls.target)

        up = (2 * meanX * meanY + c1) * (2 * sigmaX * sigmaY + c2)
        down = (meanX**2 + meanY**2 + c1) * (sigmaX + sigmaY + c2)

        ssim = up / down

        return ssim


if __name__ == "__main__":
    filePath = './Images/input/'
    fileNames = ('barbara.tif', 'cameraman.tif', 'house.tif', 'lena.tif', 'Monarch.tif',
                 'p_bar.tif', 'Parrot.tif', 'Tower.tif')

    for fileName in fileNames:
        name = filePath + fileName

        lpg_pca = LPG_PCA(name)
        lpg_pca.run()

        resource = cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        target = lpg_pca.getResult()
        cv2.imwrite('./Images/first_stage/' + fileName, target)

        scd_name = './Images/first_stage/' + fileName
        scd_lpg_pca = LPG_PCA(scd_name)
        scd_lpg_pca.run()
        scd_target = scd_lpg_pca.getResult()
        cv2.imwrite('./Images/second_stage/' + fileName, scd_target)

        be_check = Check(resource, resource)
        be_PSNR = be_check.getPSNR()
        be_SSIM = be_check.getSSIM()

        fst_check = Check(resource, target)
        fst_PSNR = fst_check.getPSNR()
        fst_SSIM = fst_check.getSSIM()

        scd_check = Check(resource, scd_target)
        scd_PSNR = scd_check.getPSNR()
        scd_SSIM = scd_check.getSSIM()

        print(fileName, end=':\n')
        print('before:')
        print('PSNR:', be_PSNR)
        print('SSIM:', be_SSIM)
        print('first:')
        print('PSNR:', fst_PSNR)
        print('SSIM:', fst_SSIM)
        print('second:')
        print('PSNR:', scd_PSNR)
        print('SSIM:', scd_SSIM, end='\n\n')
