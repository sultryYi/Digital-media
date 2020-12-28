import cv2
import numpy as np


class Harris:
    def __init__(self, filePath):
        self.grey = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        self.color = cv2.imread(filePath, cv2.IMREAD_COLOR)
        self.x = self.grey.shape[0]
        self.y = self.grey.shape[1]

    # 对整个图像求梯度
    def grad(self, ksize=3):
        # 使用Sobel梯度算子与图像矩阵做卷积，计算两个方向梯度
        Dx = cv2.Sobel(self.grey, cv2.CV_32F, 1, 0, ksize=ksize)
        Dy = cv2.Sobel(self.grey, cv2.CV_32F, 0, 1, ksize=ksize)

        return Dx, Dy

    # 高斯滤波
    def gauss(self, Dx, Dy, blockSize=3):
        # 计算cov矩阵
        cov = np.zeros((self.x, self.y, 3), dtype=np.float32)

        for i in range(self.x):
            for j in range(self.y):
                cov[i, j, 0] = Dx[i, j] * Dx[i, j]
                cov[i, j, 1] = Dx[i, j] * Dy[i, j]
                cov[i, j, 2] = Dy[i, j] * Dy[i, j]

        # 进行高斯滤波
        cov = cv2.GaussianBlur(cov, (blockSize, blockSize), 1)

        return cov

    # 计算响应值
    def getResponse(self, cov, k=0.04):
        shape0 = cov.shape[0]
        shape1 = cov.shape[1]
        res = np.zeros((shape0, shape1), dtype=np.float32)
        for i in range(shape0):
            for j in range(shape1):
                a = cov[i, j, 0]
                b = cov[i, j, 1]
                c = cov[i, j, 2]
                res[i, j] = a * c - b * b - k * (a + c) * (a + c)

        return res

    def harris(self):
        Dx, Dy = self.grad()
        cov = self.gauss(Dx, Dy)
        response = self.getResponse(cov)
        return Dx, Dy, response

    def draw(self, response, name, maxCorners=180, qualityLevel=1e-5, minDistance=30):
        pos = cv2.goodFeaturesToTrack(response, maxCorners, qualityLevel, minDistance)
        for i in range(len(pos)):
            cv2.circle(self.color, (pos[i][0][0], pos[i][0][1]), 5, [116, 255, 238], thickness=5)

        path = '../output/' + name
        cv2.imwrite(path, self.color)

        return pos


class SIFT:
    def __init__(self, img):
        self.img = img.copy()
        self.img['grey'] = cv2.imread(self.img['filePath'], cv2.IMREAD_GRAYSCALE)
        self.img['color'] = cv2.imread(self.img['filePath'], cv2.IMREAD_COLOR)
        self.img['x'] = self.img['grey'].shape[0]
        self.img['y'] = self.img['grey'].shape[1]

    def sigma(self):
        sigma = np.std(self.img['grey'])
        res = int(sigma / 10)
        return res

    def getGauss(self, size):
        kx = cv2.getGaussianKernel(size, 0)
        ky = cv2.getGaussianKernel(size, 0)
        gauss = np.multiply(kx, np.transpose(ky))
        return gauss

    def main_direction(self, sigma):
        # 角点集合，三维。
        # 第一维代表这是第几个角点
        # 第二维代表该角点
        # 第三维的[0]是列，[1]是行
        points = self.img['points']

        # 每个角点周围的梯度，半径为3*sigma
        radius = int(3 * sigma)
        size = int(2 * radius + 1)

        direction = []

        for i in range(len(points)):
            # 获得该点的坐标
            row = int(points[i][0][1])
            column = int(points[i][0][0])

            # 该点半径为radius的所有点的范围
            row_start = int(max(row - radius, 0))
            row_end = int(min(row + radius + 1, self.img['x']))

            col_start = int(max(column - radius, 0))
            col_end = int(min(column + radius + 1, self.img['y']))

            # 如果范围超出了原图像，则记下，后续补零
            row_move = int(size - (row_end - row_start))
            col_move = int(size - (col_end - col_start))
            if row_start == 0:
                row_move = int(-1 * row_move)
            if col_start == 0:
                col_move = int(-1 * col_move)

            # 计算梯度的第一范式和角度(角度取0到35的整数)
            tempDy = self.img['Dy'][row_start:row_end, col_start:col_end].copy()
            tempDx = self.img['Dx'][row_start:row_end, col_start:col_end].copy()
            tempDx[tempDx == 0] = 1e-5
            modulus_mat = np.power(np.power(tempDx,2) + np.power(tempDy,2),0.5)
            theta_mat = np.array((np.degrees(np.arctan2(tempDy, tempDx)) + 180) / 10, dtype=np.int)
            if row_move > 0:
                modulus_mat = np.r_[modulus_mat, np.zeros((row_move, col_end - col_start))]
                theta_mat = np.r_[theta_mat, np.zeros((row_move, col_end - col_start))]
            elif row_move < 0:
                modulus_mat = np.r_[np.zeros((-1 * row_move, col_end - col_start)), modulus_mat]
                theta_mat = np.r_[np.zeros((-1 * row_move, col_end - col_start)), theta_mat]
            if col_move > 0:
                modulus_mat = np.c_[modulus_mat, np.zeros((size, col_move))]
                theta_mat = np.c_[theta_mat, np.zeros((size, col_move))]
            elif col_move < 0:
                modulus_mat = np.c_[np.zeros((size, -1 * col_move)), modulus_mat]
                theta_mat = np.c_[np.zeros((size, -1 * col_move)), theta_mat]

            # 权重系数矩阵，高斯函数
            gauss = self.getGauss(size)

            # 统计
            wgt_mod = modulus_mat * gauss
            static = np.zeros((1, 36))
            for j in range(36):
                static[0][j] = np.sum(wgt_mod[theta_mat == j])

            # 保存得到的主方向
            the_max = np.max(static)
            direction.append(10 * (np.where(static == the_max)[1][0]))
            self.img['direction'] = direction

    def descriptor(self):
        points = self.img['points']
        radius = 8
        size = int(2 * radius + 1)
        self.img['descriptor'] = np.zeros((len(points), 4, 4, 8))
        for i in range(len(points)):
            # 获得该点的坐标
            row = int(points[i][0][1])
            column = int(points[i][0][0])

            # 该点半径为radius的所有点的范围
            row_start = int(max(row - radius, 0))
            row_end = int(min(row + radius + 1, self.img['x']))

            col_start = int(max(column - radius, 0))
            col_end = int(min(column + radius + 1, self.img['y']))

            # 如果范围超出了原图像，则记下，后续补零
            row_move = int(size - (row_end - row_start))
            col_move = int(size - (col_end - col_start))
            if row_start == 0:
                row_move = int(-1 * row_move)
            if col_start == 0:
                col_move = int(-1 * col_move)

            # 计算梯度的第二范式和角度(角度取0到7的整数)
            tempDy = self.img['Dy'][row_start:row_end, col_start:col_end].copy()
            tempDx = self.img['Dx'][row_start:row_end, col_start:col_end].copy()
            tempDx[tempDx == 0] = 1e-5
            modulus_mat = np.power(np.power(tempDx,2) + np.power(tempDy,2),0.5)

            theta_mat = np.array(
                np.degrees(np.arctan2(tempDy, tempDx)) + 180 - self.img['direction'][i])
            theta_mat[theta_mat < 0] = theta_mat[theta_mat < 0] + 360
            theta_mat = np.array(theta_mat / 45, dtype=np.int)

            if row_move > 0:
                modulus_mat = np.r_[modulus_mat, np.zeros((row_move, col_end - col_start))]
                theta_mat = np.r_[theta_mat, np.zeros((row_move, col_end - col_start))]
            elif row_move < 0:
                modulus_mat = np.r_[np.zeros((-1 * row_move, col_end - col_start)), modulus_mat]
                theta_mat = np.r_[np.zeros((-1 * row_move, col_end - col_start)), theta_mat]
            if col_move > 0:
                modulus_mat = np.c_[modulus_mat, np.zeros((size, col_move))]
                theta_mat = np.c_[theta_mat, np.zeros((size, col_move))]
            elif col_move < 0:
                modulus_mat = np.c_[np.zeros((size, -1 * col_move)), modulus_mat]
                theta_mat = np.c_[np.zeros((size, -1 * col_move)), theta_mat]

            # 权重系数矩阵，高斯函数
            gauss = self.getGauss(size)

            wgt_mod = modulus_mat * gauss

            # 获得每个点对应的128维向量，保存为一个(4, 4, 8)的array
            for j in range(4):
                for k in range(4):
                    # anchor为计算的4*4小区域左上角点的坐标
                    anchor = [4 * j, 4 * k]
                    if j >= 2:
                        anchor[0] = anchor[0] + 1
                    if k >= 2:
                        anchor[1] = anchor[1] + 1

                    # focus_mod为4*4小区域的一范式
                    # focus_theta为4*4小区域的角度
                    static = np.zeros((1, 8))
                    focus_mod = wgt_mod[anchor[0]:anchor[0] + 4, anchor[1]:anchor[1] + 4].copy()
                    focus_theta = theta_mat[anchor[0]:anchor[0] + 4, anchor[1]:anchor[1] + 4].copy()

                    # 分别求出八个方向的第一范式和
                    for theta in range(8):
                        static[0][theta] = np.sum(focus_mod[focus_theta == theta])

                    # 将这个8维向量保存起来
                    self.img['descriptor'][i][j][k] = static[0]

            # 消除亮度影响
            # 对每个128维向量，分别做一次归一化(化为单位向量)
            vector128 = self.img['descriptor'][i].copy()
            self.img['descriptor'][i] = vector128 / np.linalg.norm(vector128)
            # 然后将大于0.2的值变为0.2
            self.img['descriptor'][i][self.img['descriptor'][i] > 0.2] = 0.2
            # 再做一次归一化
            vector128 = self.img['descriptor'][i].copy()
            self.img['descriptor'][i] = vector128 / np.linalg.norm(vector128)

        # 将(4, 4, 8)的array化为一个128维向量
        self.img['descriptor'] = self.img['descriptor'].reshape((len(points), -1))

    def calculate(self):
        self.main_direction(self.sigma())
        self.descriptor()


class Match:
    def __init__(self, img1, img2):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        self.match_point = []

    def match(self):
        for i in range(len(self.img1['descriptor'])):
            distance = np.sum(np.power(self.img2['descriptor'] - self.img1['descriptor'][i], 2),
                              axis=1)

            if np.min(distance) >= 0.2:
                continue
            
            j = np.where(distance == np.min(distance))[0][0]

            self.match_point.append(
                (self.img1['points'][i][0], self.img2['points'][j][0], np.min(distance)))
        self.match_point.sort(key=self.takeThird)

    def takeThird(self,elem):
        return elem[2]

    def draw(self):
        self.match()
        color_img1 = self.img1['color'].copy()
        color_img2 = self.img2['color'].copy()
        pos1 = self.img1['points'].copy()
        pos2 = self.img2['points'].copy()
        lines = self.match_point.copy()

        for i in range(len(pos1)):
            cv2.circle(color_img1, (pos1[i][0][0], pos1[i][0][1]), 5, [116, 255, 238], thickness=5)
        for i in range(len(pos2)):
            cv2.circle(color_img2, (pos2[i][0][0], pos2[i][0][1]), 5, [116, 255, 238], thickness=5)
        color_all = np.hstack((color_img1, color_img2))
        for i in range(len(lines)):
            cv2.line(color_all, (lines[i][0][0], lines[i][0][1]),
                     (int(lines[i][1][0] + color_img1.shape[1]), lines[i][1][1]), (133, 255, 116),
                     3)

        path = '../output/match.bmp'
        cv2.imwrite(path, color_all)


if __name__ == "__main__":
    img1 = {'filePath': '../src/1.bmp', 'name': '1.bmp'}
    img2 = {'filePath': '../src/2.bmp', 'name': '2.bmp'}
    imgs = [img1, img2]
    corners = []
    sifts = []

    for img in imgs:
        corner = Harris(img['filePath'])
        img['Dx'], img['Dy'], response = corner.harris()
        img['points'] = corner.draw(response, img['name'])

    for img in imgs:
        sift = SIFT(img)
        sift.calculate()
        sifts.append(sift)

    match = Match(sifts[0].img, sifts[1].img)
    match.draw()
