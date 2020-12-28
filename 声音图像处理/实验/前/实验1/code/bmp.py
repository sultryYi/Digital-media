from struct import unpack
import numpy as np
import cv2


class Bitmap:
    def __init__(self, filePath):
        self.fileHeader = self.BITMAPFILEHEADER(filePath)
        self.infoHeader = self.BITMAPINFOHEADER(filePath)

        self.height = self.infoHeader.biHeight
        self.width = self.infoHeader.biWidth
        bfOffBits = self.fileHeader.bfOffBits

        row = self.height
        column = self.width
        if (self.width * 3 % 4) != 0:
            self.placeholder = 4 - (self.width * 3 % 4)
        else:
            self.placeholder = 0
        self.img = np.ndarray((row, column), dtype=tuple)

        with open(filePath, 'rb') as bitmap:
            temp = bitmap.read(bfOffBits)
            for x in range(row):
                for y in range(column):
                    b = unpack('<B', bitmap.read(1))[0]
                    g = unpack('<B', bitmap.read(1))[0]
                    r = unpack('<B', bitmap.read(1))[0]
                    self.img[row - 1 - x][y] = (b, g, r)
                if (self.placeholder != 0):
                    temp = bitmap.read(self.placeholder)
        self.B, self.G, self.R = self.departBGR(self.img, row, column)

    def departBGR(self, img, row, column):
        B = np.zeros_like(img, dtype=np.uint8)
        G = np.zeros_like(img, dtype=np.uint8)
        R = np.zeros_like(img, dtype=np.uint8)
        for x in range(row):
            for y in range(column):
                BGR = img[x][y]
                B[x][y] = BGR[0]
                G[x][y] = BGR[1]
                R[x][y] = BGR[2]
        return B, G, R

    def getPixel(self):
        print('以左上角为原点，获得某像素点的RGB值:(R,G,B)')
        print('x为0至%d的整数，y为0至%d的整数' % (self.width - 1, self.height - 1))
        while True:
            x = int(input('请输入x:'))
            if x < 0 or x >= self.width:
                print('非法的x')
                continue
            y = int(input('请输入y:'))
            if y < 0 or y >= self.height:
                print('非法的y')
                continue
            print('像素点(%d,%d)的RGB值为(%d,%d,%d)' %
                  (x, y, self.R[x][y], self.G[x][y], self.B[x][y]))
            break

    def writeImg(self):
        b = self.B
        g = self.G
        r = self.R
        merged = cv2.merge([b, g, r])
        cv2.imwrite("output/fav.bmp", merged)

    def showImg(self):
        b = self.B
        g = self.G
        r = self.R
        merged = cv2.merge([b, g, r])
        cv2.imshow('any key to continue', merged)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def printHeaderInfo(self):
        print('BITMAPFILEHEADER')
        print('bfType:', self.fileHeader.bfType)
        print('bfSize:', self.fileHeader.bfSize)
        print('bfReserved1:', self.fileHeader.bfReserved1)
        print('bfReserved2:', self.fileHeader.bfReserved2)
        print('bfOffBits:', self.fileHeader.bfOffBits)
        print('\nBITMAPINFOHEADER')
        print('biSize:', self.infoHeader.biSize)
        print('biWidth:', self.infoHeader.biWidth)
        print('biHeight:', self.infoHeader.biHeight)
        print('biPlanes:', self.infoHeader.biPlanes)
        print('biBitcount:', self.infoHeader.biBitcount)
        print('biCompression:', self.infoHeader.biCompression)
        print('biSizeImage:', self.infoHeader.biSizeImage)
        print('biXPelsPerMeter:', self.infoHeader.biXPelsPerMeter)
        print('biYPelsPerMeter:', self.infoHeader.biYPelsPerMeter)
        print('biClrUsed:', self.infoHeader.biClrUsed)
        print('biClrImportant:', self.infoHeader.biClrImportant)

    class BITMAPFILEHEADER:
        def __init__(self, filePath):
            with open(filePath, 'rb') as bitmap:
                #         bfType:2字节，说明文件类型，一般为19778，其转化为十六进制为0x4d42，对应的字符串为BM
                #         bfSize:4字节，文件大小，以字节为单位
                #         bfReserved1:2字节，保留，为0
                #         bfReserved1:2字节，保留，为0
                #         bfOffBits:4字节，从文件开始处到像素数据的偏移，也就是这两个结构体大小之和
                self.bfType = unpack('<h', bitmap.read(2))[0]
                self.bfSize = unpack('<i', bitmap.read(4))[0]
                self.bfReserved1 = unpack('<h', bitmap.read(2))[0]
                self.bfReserved2 = unpack('<h', bitmap.read(2))[0]
                self.bfOffBits = unpack('<i', bitmap.read(4))[0]

    class BITMAPINFOHEADER:
        def __init__(self, filePath):
            with open(filePath, 'rb') as bitmap:
                bitmap.read(14)
                #         bisize:4字节，此信息头大小
                #         biWidth:4字节，图像的宽
                #         biHeight:4字节，图像的高，正数代表位图为倒向，复数代表位图为正向，通常为正数
                #         biPlanes:2字节，图像的帧数，一般为1
                #         biBitcount:2字节，一像素所占的位数，一般为24
                #         biCompression:4字节，说明图像数据压缩类型，一般为0(不压缩)
                #         biSizeImage:4字节，像素数据所占大小，说明图像的大小，以字节为单位。当压缩类型为0时，总设置为0
                #         biXPelsPerMeter:4字节，水平分辨率，用像素/米表示，有符号整数
                #         biYPelsPerMeter:4字节，水平分辨率，用像素/米表示，有符号整数
                #         biClrUsed:4字节，说明位图实际使用的彩色表中的颜色索引数，若设为0则说明使用所有调色板项
                #         biClrImportant:4字节，说明对图像显示有重要影响的颜色索引的数目。若为0，表示都重要
                self.biSize = unpack('<i', bitmap.read(4))[0]
                self.biWidth = unpack('<i', bitmap.read(4))[0]
                self.biHeight = unpack('<i', bitmap.read(4))[0]
                self.biPlanes = unpack('<h', bitmap.read(2))[0]
                self.biBitcount = unpack('<h', bitmap.read(2))[0]
                self.biCompression = unpack('<i', bitmap.read(4))[0]
                self.biSizeImage = unpack('<i', bitmap.read(4))[0]
                self.biXPelsPerMeter = unpack('<i', bitmap.read(4))[0]
                self.biYPelsPerMeter = unpack('<i', bitmap.read(4))[0]
                self.biClrUsed = unpack('<i', bitmap.read(4))[0]
                self.biClrImportant = unpack('<i', bitmap.read(4))[0]


# 读bmp图像
filePath = 'bmp/fav.bmp'
bitmap = Bitmap(filePath)
# 打印文件头的信息头的信息
bitmap.printHeaderInfo()
# 展示该图像
bitmap.showImg()
# 将读到的bmp图像重新写入到output文件夹中
bitmap.writeImg()
# 获取某点的RGB值
bitmap.getPixel()


filePath = 'bmp/fav.bmp'

def slicing(filePath):
    img = cv2.imread(filePath)
    (y, x, alpha) = img.shape
    
    while True:
        chooseX = input('请输入选区宽度，不输入为全图%d像素:'%(x))
        if chooseX != '' and (int(chooseX) > x or int(chooseX) <= 0):
            print('输入非法！')
            continue
        chooseY = input('请输入选取高度，不输入为全图%d像素:'%(y))
        if chooseY != '' and (int(chooseY) > y or int(chooseY) <= 0):
            print('输入非法！')
            continue
        chooseM = int(input('请输入图块宽度:'))
        if chooseM > x or chooseM <= 0:
            print('输入非法！')
            continue
        chooseN = int(input('请输入图块高度:'))
        if chooseN > y or chooseN <= 0:
            print('输入非法！')
            continue
        break
    if chooseX != '':
        x = int(chooseX)
    if chooseY != '':
        y = int(chooseY)
    
    m = chooseM
    n = chooseN
    
    atomX = m
    atomY = n
    slicingY = int(y / atomY)
    slicingX = int(x / atomX)
    #     shuffleY = slicingY
    #     shuffleX = slicingX
    #     if y % atomY != 0:
    #         slicingY += 1
    #     if x % atomX != 0:
    #         slicingX += 1
    nowX = 0
    nowY = 0
    nextX = min(nowX + atomX, x)
    nextY = min(nowY + atomY, y)
    matrix = np.zeros((slicingY, slicingX), dtype=list)
    for i in range(slicingY):
        for j in range(slicingX):
            matrix[i][j] = img[nowY:nextY, nowX:nextX].copy()
            nowX = nowX + atomX
            nextX = min(nowX + atomX, x)
        nowY = nowY + atomY
        nextY = min(nextY + atomY, y)
        nowX = 0
        nextX = min(nowX + atomX, x)


#     cv2.imshow('slicing',matrix[6][9])
#     cv2.moveWindow('slicing',400,400)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

#     shuffleMatrix = np.random.permutation(matrix[0:shuffleY, 0:shuffleX])
    preShuffleMatrix = matrix[0:slicingY, 0:slicingX].copy()
    shuffleMatrix = preShuffleMatrix.copy()
    sort = np.random.permutation(np.array(range(slicingY * slicingX)))
    sort = sort.reshape((slicingY, slicingX))
    for i in range(slicingY):
        for j in range(slicingX):
            index = sort[i][j]
            shuffleMatrix[i][j] = preShuffleMatrix[int(
                index / slicingX)][index % slicingX]

    shuffle = img.copy()
    for i in range(slicingY * atomY):
        for j in range(slicingX * atomX):
            for k in range(alpha):
                shuffle[i][j][k] = shuffleMatrix[int(i / atomY)][int(
                    j / atomX)][i % atomY][j % atomX][k]
    cv2.imwrite('output/shuffle.bmp', shuffle)
    
    cv2.imshow('any key to continue',shuffle)
    cv2.waitKey()
    cv2.destroyAllWindows()

slicing(filePath)