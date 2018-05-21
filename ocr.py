# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib
#matplotlib.use('WXagg')
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
#提取峰值函数，提取数组里面的峰值，然后找出文本行
def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2): #有些图片比较多噪音，需要minimun_val和minimun_range用于过滤噪音。
    start_i = None
    end_i = None
    peek_ranges = []
    #print(array_vals)
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:   #不大于这个最小范围证明是噪音
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


img = cv2.imread('32.png')

roi=img[0:80,0:1000]

GrayImage=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #将图片变为灰度图

##去除黑色的水印
rows,cols=GrayImage.shape
for i in range(rows):
    for j in range(cols):
        if GrayImage[i, j] == 0:  # 0代表黑色的点
            GrayImage[i,j]=255  # 如果是黑色，则把它变成白色
ret,mask_bin = cv2.threshold(GrayImage,70,255,cv2.THRESH_BINARY) #阈值化操作，得到二值图像mask_bin


adaptive_threshold = cv2.bitwise_not(mask_bin) #反黑白掩膜

roi=adaptive_threshold
#adaptive_threshold = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)


#print(adaptive_threshold.shape)
#提取每行
horizontal_sum = np.sum(roi, axis=1) #把每一行的像素值都加起来，原来是200行1200列，加完之后就是200行，每一行的值都是原来1200列的和
#画出行的图
plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
plt.gca().invert_yaxis()
plt.show()

peek_ranges = extract_peek_ranges_from_array(horizontal_sum) #用自定义的提取峰值函数，提取数组里的峰值 得到的结果是有像素值也就是有字部分的位置的范围

###画出切割出来的矩形
line_seg_adaptive_threshold = np.copy(roi)

for i, peek_range in enumerate(peek_ranges):
    x = 0
    y = peek_range[0]
    w = line_seg_adaptive_threshold.shape[1]
    h = peek_range[1] - y
    pt1 = (x, y) #所画矩形左上角的点
    pt2 = (x + w, y + h)
    cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
cv2.imshow('line image', line_seg_adaptive_threshold)
#cv2.waitKey(0)


##竖着切，切出来单个字
vertical_peek_ranges2d = []
for peek_range in peek_ranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = roi[start_y:end_y, :] #line_img为切割出来的每一行
    vertical_sum = np.sum(line_img, axis=0)#对每一行的像素值按列求和
    print("%s",1,vertical_sum)
    vertical_peek_ranges = extract_peek_ranges_from_array(vertical_sum,minimun_val=900,minimun_range=1)
    vertical_peek_ranges2d.append(vertical_peek_ranges)


vertical = np.copy(roi)
## Draw
color = (255, 255, 255)
for i, peek_range in enumerate(peek_ranges):
    for vertical_range in vertical_peek_ranges2d[i]:
        x = vertical_range[0]
        y = peek_range[0]
        w = vertical_range[1] - x
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(vertical, pt1, pt2, color)
cv2.imshow('char image', vertical)
cv2.waitKey(0)



