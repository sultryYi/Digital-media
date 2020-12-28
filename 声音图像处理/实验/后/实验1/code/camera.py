# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:05:07 2020

@author: 徐满阳
"""
import queue
import threading
import cv2
import subprocess as sp
import datetime,time
import math
import tkinter
import matplotlib.pyplot as plt
#import sys
from playsound import playsound
#import matplotlib.pyplot as plt
#配置文件
protoFile = './models/pose/coco/pose_deploy_linevec.prototxt'
weightsfile = './models/pose/coco/pose_iter_440000.caffemodel'
#face_cascade = cv2.CascadeClassifier('./models/face.xml')
#eye_cascade = cv2.CascadeClassifier('./models/eye.xml')
npoints = 18
POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
#加载网络
net = cv2.dnn.readNetFromCaffe(protoFile,weightsfile)
#读取图片
def _readImage(file,num):
    #im = cv2.imread('./images/'+file+'.jpg')
    #（1440*1080）
    im = cv2.cvtColor(file,cv2.COLOR_BGR2RGB)
    inHeight = im.shape[0]  #1440
    inWidth = im.shape[1]   #1080
    netInputsize = (368,368)
    #转为inpBlob格式
    inpBlob = cv2.dnn.blobFromImage(im,1.0/255,netInputsize,(0,0,0),swapRB=True,crop=False)
    #归一化后的图片最为输入传入net网络,然后输出
    net.setInput(inpBlob)
    output = net.forward()  #1*57*46*46
    scaleX = float(inWidth) / output.shape[3]    #float(1080)/64 = 23.47826086956522
    scaleY = float(inHeight)/ output.shape[2]    #float(1440)/64 = 31.304347826086957
    points = []
    threshold = 0.1
    tantou=0
    boolean=0
    for i in range(npoints):
        probMap = output[0,i,:,:]  #shape(46*46)
        minVal,prob,minLoc,point =cv2.minMaxLoc(probMap)
        x = scaleX * point[0]
        y = scaleY * point[1]
        if prob > threshold:
            points.append((int(x),int(y)))
        else:
            if i==16:
                tantou=17
            elif i==17:
                tantou=16
            points.append(None)
        #points[]最后为18个关键点的坐标
        #[(516, 313), (516, 438), (399, 438), (375, 626), (352, 751), (610, 438), (633, 594), (657, 751), (446, 751),
        # (446, 970), (446, 1158), (563, 782), (540, 1001), (540, 1064), (493, 281), (540, 281), (446, 313), (563, 313)]

    imPoints = im.copy()
    #imSkeleton = im.copy()
    for i,p in enumerate(points):
        #enumerate把points的值前面带上索引i
        if points[i]!=None:
            cv2.circle(imPoints,p,8,(255,255,0),thickness=1,lineType=cv2.FILLED)
            cv2.putText(imPoints,'{}'.format(i),p,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,lineType=cv2.LINE_AA)
    """
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        #if points[partA] and points[partB]:
        cv2.line(imSkeleton,points[partA],points[partB],(255, 255,0),2)
        cv2.circle(imSkeleton, points[partA],8,(255,0,0),thickness=-1, lineType=cv2.FILLED)
    """
    music='tantou.mp3'
    info='请勿交头接耳'
    #检测探头
    if tantou!=0 and points[tantou]!=None and points[1]!=None:
        dy=abs(points[tantou][0]-points[1][0])
        dx=abs(points[tantou][1]-points[1][1])
        angle=math.atan2(dy,dx)
        #print(points[tantou],points[0])
        angle=int(angle*180/math.pi)
        #print(file,angle,dx,dy)
        if angle<15:
            #print(num,'请勿交头接耳',datetime.datetime.now())#tkinter.messagebox.showwarning('警告','请勿交头接耳')#
            #threading.Thread(target=playsound, args=("tantou.mp3",)).start()
            boolean=1
    count0=0
    if points[0]!=None:count0+=1
    for i in range(14, 18):
        if points[i]!=None:count0+=1
    count1=0
    for i in range(1, 8):
        if points[i]!=None:count1+=1
    #if (points[0]==None or points[14]==None or points[15]==None)and(points[1]!=None or points[2]!=None or points[5]!=None):
    if count0<3 and count1>=2:
        #print(num,'别睡了，起来嗨',datetime.datetime.now())#tkinter.messagebox.showinfo('提示','别睡了起来嗨')#
        #threading.Thread(target=playsound, args=("tantou.mp3",)).start()
        #th.setDaemon(True)
        #th.start()
        music='tantou.mp3'
        info='别睡了起来嗨'
        boolean=1
    #if points[0]==None or points[1]==None or points[2]==None or points[5]==None or points[14]==None or points[15]==None or points[16]==None or points[17]==None:
    if count0<3 and count1<2:
        #print(num,'人呢',datetime.datetime.now())#tkinter.messagebox.showinfo('提示','人呢')#
        #threading.Thread(target=playsound, args=("yinxiao.mp3",)).start()
        #th.setDaemon(True)
        #th.start()
        music='yinxiao.mp3'
        info='人呢'
        boolean=1
    if points[4]==None and points[7]==None:
        #print(num,'手呢',datetime.datetime.now())#tkinter.messagebox.showinfo('提示','手呢')#
        #threading.Thread(target=playsound, args=("yinxiao.mp3",)).start()
        #th.setDaemon(True)
        #th.start()
        music='yinxiao.mp3'
        info='手呢'
        boolean=1
    #ths=[
        #threading.Thread(target=playsound, args=(music,)),
        #threading.Thread(target=tkinter.messagebox.showinfo,args=('提示',info))]
    #[thread.setDaemon(True) for thread in ths]
    #[thread.start() for thread in ths]
    if boolean==1:
        #cv2.putText(imPoints,'别睡了',(100,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,lineType=cv2.LINE_AA)
        print(num,info,datetime.datetime.now())
        threading.Thread(target=playsound, args=(music,)).start()
        plt.figure('提示：'+info,figsize=(3,2))
        plt.axis('off')#;plt.imshow(imPoints)
        #plt.title(r'注意:'+info,fontproperties='SimHei',fontsize=25)
        plt.show()
        cv2.imwrite('./images/'+'res'+str(num)+'.jpg',imPoints)
        time.sleep(2)
    #else:
        #cv2.imwrite('./images/res/'+'res'+str(num)+'.jpg',imPoints)
    #plt.subplot(121)
    #plt.subplot(122)
    #plt.axis('off');plt.imshow(imSkeleton)
    #cv2.imwrite('D:/openpose-master/images/'+'res'+str(file)+'.jpg',imPoints)
#def showInfo(string):
    #tkinter.messagebox.showinfo('提示',string)
class Live(object):
    def __init__(self):
        self.frame_queue = queue.Queue()
        #self.frame_queueC = queue.Queue()
        self.frame_queueS = queue.Queue()
        self.command = ""
        # 自行设置
        self.rtmpUrl = "rtmp://106.54.116.250:1935/live/hwtadie"
        self.camera_path = 0
        self.count=0
        self.cap = cv2.VideoCapture(self.camera_path)
        self.cap.set(3,640)
        self.cap.set(4,480)
        self.start=datetime.datetime.now()
        self.end=datetime.datetime.now()
        #self.start1=datetime.datetime.now()
        #self.end1=datetime.datetime.now()
        #self.bool=0
        self.flag=0
    def read_frame(self):
        print("开启推流")
        # Get video information
        fps = 25#int(cap.get(cv2.CAP_PROP_FPS))
        width = 640#int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = 480#int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # ffmpeg command
        self.command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                self.rtmpUrl]

        # read webcamera
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                print("Opening camera is failed")
                # 说实话这里的break应该替换为：
                # cap = cv2.VideoCapture(self.camera_path)
                # 因为我这俩天遇到的项目里出现断流的毛病
                # 特别是拉取rtmp流的时候！！！！
                break
            # put frame into queue
            #if self.bool==0:self.bool=1
            self.frame_queue.put(frame)
            self.frame_queueS.put(frame)#cv2.imshow("frame",frame)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
            #faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            #if len(faceRects) > 0:  # 大于0则检测到人脸
                #for faceRect in faceRects:  # 单独框出每一张人脸
                    #x, y, w, h = faceRect
                    #cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            # 展示结果
            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                #break
            # process frame
            # 你处理图片的代码
            self.end=datetime.datetime.now()
            if self.count==0 or (self.end-self.start).seconds>=3:
               print(self.count,'-----捕捉照片-----',datetime.datetime.now())
               self.start=self.end
               th=threading.Thread(target=_readImage, args=(frame,str(self.count)))#self.frame_queueC.put(frame)#cv2.imwrite('D:/openpose-master/images/'+str(self.count)+'.jpg',frame)
               th.setDaemon(True)
               th.start()
               self.count+=1#print(self.count,datetime.datetime.now())#print('2--',self.start,self.end)

    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break
        while True:
            if self.frame_queue.empty() != True:
                frame = self.frame_queue.get()
                p.stdin.write(frame.tostring())
    def show(self):
        #time.sleep(1)
        while True:
            if self.frame_queueS.empty()!=True:
                frame=self.frame_queueS.get()
                '''
                self.end1=datetime.datetime.now()
                if (self.end1-self.start1).seconds>=3:
                #print('1--',self.start,self.end)
                    self.start1=self.end1
                    faces=face_cascade.detectMultiScale(frame, 1.3, 2)
                    #img=frame
                    if len(faces)==0:
                        print('脸呢')
                        playsound('tantou.mp3')
                    for (x,y,w,h) in faces:
                        #frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        face_area=frame[y:y+h,x:x+w]
                        eyes = eye_cascade.detectMultiScale(face_area,1.3,10)
                        if len(eyes)==0:
                            print('眼呢')
                            playsound('yinxiao.mp3')
                        #for (ex,ey,ew,eh) in eyes:
                        #画出人眼框，绿色，画笔宽度为1
                            #cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                    '''
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                #if len(faceRects) > 0:  # 大于0则检测到人脸
                    #for faceRect in faceRects:  # 单独框出每一张人脸
                        #x, y, w, h = faceRect
                        #cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
                #else:
                    #print('快回来考试')
                    #playsound("tantou.mp3")
                cv2.imshow("capture",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    def handle(self,delay):
        count=0
        while True:
            time.sleep(delay)
            if self.frame_queueC.empty()!=True:
                print(count,'开始处理照片',datetime.datetime.now())
                th=threading.Thread(target=_readImage, args=(self.frame_queueC.get(),str(count)))
                th.setDaemon(True)
                th.start()
                #print('-------------',count,datetime.datetime.now())
                #_readImage(self.frame_queueC.get(),str(count))
            count+=1
            if self.flag==1:
                print('停止处理')
                break
    def warning(self):
        for i in range(10):
            time.sleep(30)
            playsound('time.mp3')
            if self.flag==1:
                print('停止计时')
                break
    def run(self):
        print("开启线程")
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.push_frame, args=(self,)),
            #threading.Thread(target=Live.handle, args=(self,4)),
            threading.Thread(target=Live.warning, args=(self,)),
            threading.Thread(target=Live.show, args=(self,))
        ]
        [thread.setDaemon(True) for thread in threads]
        self.start=datetime.datetime.now()
        #self.start1=datetime.datetime.now()
        [thread.start() for thread in threads]
        '''
        while True:
            if self.bool==1:
                if self.frame_queueS.empty()!=True:
                    cv2.imshow("监控",self.frame_queueS.get())
            if flag==1:
                cv2.destroyAllWindows()
                break
        '''
live=Live()
live.run()
print('按0结束')
string=input()
if string=='0':
    live.flag=1
    live.cap.release()
    cv2.destroyAllWindows()
else:
    print(string)

