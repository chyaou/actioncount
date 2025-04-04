import sys, csv, cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction,
                             QFileDialog, QMessageBox)
from UI.Ui_ui1 import Ui_MainWindow
import mediapipe as mp
import numpy as np
import computer
from body import pre_image


class Main(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initMenuBar()
        self.button()
        self.default()

    def initMenuBar(self):  # 菜单栏
        menuBar = self.menuBar()
        menuFile = menuBar.addMenu('文件(&F)')
        actionExit = QAction('退出(&Q)', self, shortcut='Ctrl+Q', triggered=QApplication.instance().quit)
        menuFile.addAction(actionExit)

    def default(self):  # 一些默认值
        self.filename = []
        self.keypoint = {0: [(23, 11, 23, 25), (24, 12, 24, 26)], 1: [(13, 11, 13, 15), (14, 12, 14, 16)],
                         2: [(13, 11, 13, 15), (14, 12, 14, 16)],
                         3: [(26, 24, 26, 28), (25, 23, 25, 27)]}  # 0 躯干大腿角[L,R], 1 肘关节角[L,R]，2......以此类推
        self.angle = {0: [90, 100], 1: [110, 140], 2: [90, 140],
                      3: [90, 140]}  # 角度范围 0仰卧起坐躯干-大腿角度, 1俯卧撑肘关节角度，2引体向上......以此类推
        self.countBtn = False

    def button(self):  # 按钮
        self.pushButton.clicked.connect(self.video)
        self.pushButton_2.clicked.connect(self.cam)
        self.pushButton_3.clicked.connect(self.analyze)
        self.pushButton_4.clicked.connect(self.btn)

    def video(self):  # 选择视频
        if self.comboBox.currentIndex() != 4:
            self.filename = QFileDialog.getOpenFileNames(self, '选择相同动作视频', '', '')[0]
            self.listWidget.clear()
            self.listWidget.addItems(self.filename)
            print(type(self.filename))
        else:
            self.filename2 = QFileDialog.getOpenFileNames(self, '选择动作视频', '', '')[0]
            self.listWidget.clear()
            self.listWidget.addItems(self.filename2)
            print(self.filename2)

    def analyze(self):
        if self.comboBox.currentIndex() != 4:  # 开始计数
            fileName = QFileDialog.getSaveFileName(self, 'Save File', 'result', "CSV Files (*.csv)")
            if fileName[0]:
                act = self.comboBox.currentIndex()
                joint = self.keypoint[act]
                angleRange = self.angle[act]
                if self.filename != []:
                    with open(fileName[0], 'w', newline='') as file0:
                        write = csv.writer(file0)
                        write.writerow(['名称', '数量'])
                        self.countBtn = True
                        for file in self.filename:
                            name = file.split('/')[-1][:-4]  # 视频文件名
                            print('当前视频：' + name)
                            count = self.count(file, joint, angleRange)
                            print(count)
                            write.writerow([name, count])
                        saveTxt = '数据文件已保存至' + str(fileName[0])
                        QMessageBox.information(self, '完成', saveTxt)
                        self.countBtn = False
        else:
            cap = cv2.VideoCapture(self.filename2[0])
            while True:
                ret, frame = cap.read()
                if ret:
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imshow('live', pre_image(frame))
                    if cv2.waitKey(1) & 0xFF == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()

    def cam(self):
        url = 0
        cap = cv2.VideoCapture(0)
        cap.release()
        cv2.destroyAllWindows()
        act = self.comboBox.currentIndex()
        cap = cv2.VideoCapture(0)
        if act != 4:
            joint = self.keypoint[act]
            angleRange = self.angle[act]
            self.count(url, joint, angleRange)
        else:
            while True:
                ret, frame = cap.read()
                if ret:
                    if self.countBtn is True:
                        frame = pre_image(frame)
                    cv2.imshow('Press ESC to Exit', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    break
        cap.release()
        cv2.destroyAllWindows()

    def btn(self):
        if self.countBtn:
            self.countBtn = False
            self.pushButton_4.setText('开始识别')
        else:
            self.countBtn = True
            self.pushButton_4.setText('停止识别')

    def count(self, video, point, angleRange):  # video视频路径， point(0,1,2,3)前4个关键点， range关节角度范围
        global count
        mp_drawing = mp.solutions.drawing_utils
        myPose = mp.solutions.pose
        poses = myPose.Pose()
        cap = cv2.VideoCapture(video)
        if self.comboBox.currentIndex() != 4:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.2)  # 视频宽度*0.2  用于显示计数
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.2)  # 视频高度*0.2
            size = int(height / 30)  # 字体大小
            count = 0  # 计数
            down = False
            up = False
            while True:
                ret, frame = cap.read()
                if ret:
                    if self.countBtn is True:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = poses.process(frame)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if results.pose_landmarks:  # 包含姿势地标的“姿势地标”字段。
                            landmarks = results.pose_landmarks.landmark
                            data = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks])  # 获取坐标
                            try:
                                # 选择身体左或右侧
                                left = point[0]  # (13,11,13,15)点序号
                                right = point[1]
                                leftP = 0
                                rightP = 0
                                for i in range(4):  # 计算能见度，哪边能见度最高，计算哪边的角度
                                    leftP += data[left[i]][3]
                                    rightP += data[right[i]][3]
                                if leftP <= rightP:
                                    value = right
                                else:
                                    value = left
                                # 计算身体角度
                                joint = round(
                                    computer.cal_angle(data[value[0]], data[value[1]], data[value[2]], data[value[3]]),
                                    2)  # v1上端中，v2上远，v3下中，v4下远
                                # 判断并计数
                                if joint > angleRange[1]:  # 从上到下，一个过长表示一次
                                    down = True
                                elif down is True and joint < angleRange[0]:
                                    up = True
                                if down is True and up is True:
                                    count += 1
                                    down = False
                                    up = False
                                # print(joint)
                                # print(count)
                            except Exception as e:
                                print(e.args)
                            # 画图
                            mp_drawing.draw_landmarks(frame, results.pose_landmarks, myPose.POSE_CONNECTIONS)
                            cv2.putText(frame, str(count), (width, height), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 255),
                                        3)
                    cv2.imshow('Press ESC to Exit', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    break
        else:
            pass
        cap.release()
        cv2.destroyAllWindows()
        return count


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())

