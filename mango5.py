# 轮廓提取、属性、近似轮廓、边界矩形和外接圆
import cv2 as cv


# 转二进制图像
def ToBinray():
    global imgray, binary
    # 1、灰度图
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('imgray', imgray)

    # 2、二进制图像
    ret, binary = cv.threshold(imgray, 127, 255, 0)
    # 阈值 二进制图像
    cv.imshow('binary', binary)


# 提取轮廓
def GetContours():
    global contours
    # 1、根据二值图找到轮廓
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 轮廓      层级                               轮廓检索模式(推荐此)  轮廓逼近方法

    # 2、画出轮廓
    dst = img.copy()
    dst = cv.drawContours(dst, contours, -1, (0, 0, 255), 3)
    #                           轮廓     第几个(默认-1：所有)   颜色       线条厚度

    cv.imshow('contours', dst)


# 获取轮廓信息
def GetContours_Attrib():
    # 画出第一个轮廓
    cnt = contours[0]
    dst = img.copy()
    dst = cv.drawContours(dst, cnt, -1, (0, 0, 255), 3)
    cv.imshow('contour0', dst)

    # 获取轮廓面积
    area = cv.contourArea(cnt)
    print("轮廓面积：", area)

    # 周长（True表示合并）
    perimeter = cv.arcLength(cnt, True)
    print("轮廓周长：", perimeter)


# 轮廓近似
def GetApprox():
    # 1、取外围轮廓
    cnt = contours[0]

    # 2、设置精度（从轮廓到近似轮廓的最大距离）
    epsilon = 0.01 * cv.arcLength(cnt, True)
    #                            轮廓  闭合轮廓还是曲线

    # 3、获取近似轮廓
    approx = cv.approxPolyDP(cnt, epsilon, True)
    #                             近似度(这里为5%)   闭合轮廓还是曲线

    # 4、绘制轮廓
    dst = img.copy()
    dst = cv.drawContours(dst, [approx], -1, (0, 0, 255), 3)

    # 显示
    cv.imshow("apporx", dst)


# 获取边界矩形
def BoundingRect():
    # 1、取外围轮廓
    cnt = contours[0]

    # 2、获取正方形坐标长宽
    x, y, w, h = cv.boundingRect(cnt)

    # 3、画出矩形
    dst = img.copy()
    dst = cv.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # 显示
    cv.imshow("rect", dst)


# 获取外接圆
def Circle():
    # 1、获取第一个轮廓
    cnt = contours[0]

    # 2、获取外接圆
    (x, y), radius = cv.minEnclosingCircle(cnt)
    # 坐标   半径

    # 3、画圆
    dst = img.copy()
    dst = cv.circle(dst, (int(x), int(y)), int(radius), (0, 0, 255), 3)

    # 显示
    cv.imshow("circle", dst)


if __name__ == '__main__':
    img = cv.imread('output/azure_rgb_5fps_2500expo_001.png')
    cv.imshow('img', img)

    ToBinray()  # 转二进制

    GetContours()  # 提取轮廓

    GetContours_Attrib()  # 获取轮廓信息

    GetApprox()  # 轮廓近似

    BoundingRect()  # 边界矩形

    Circle()  # 外接圆

    cv.waitKey(0)