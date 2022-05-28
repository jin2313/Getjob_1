import json
import threading
from datetime import time

from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.db.models import Max, Count
from django.shortcuts import render
from interviewapp.models import Question, Result
from tensorflow.keras.utils import img_to_array   # 빨간줄 떠도 작동함
from keras.models import load_model

import imutils
import os
import speech_recognition as sr
import cv2
import numpy as np


# Create your views here.
def ThresView(request):
    if request.method == 'GET':
        return render(request, 'interviewapp/threshold.html')
    else:
        print(request.POST)
        global threshold
        threshold = int(request.POST['threshold'])
        return render(request, 'interviewapp/threshold.html')



def QuestionView(request):
    if request.method == 'GET':
        corp_name = request.GET.get('corp', None)
        dept_name = request.GET.get('dept', None)
        quest_id = request.GET.get('question', None)
        next_id = str(int(quest_id) + 1)
        question = Question.objects.filter(quest_id=quest_id)
        quest_level = str(question[0].level) if len(question) != 0 else '1'
        result = Result.objects.filter(user_id=request.user)
        if len(result) == 0:
            report_num = 1
        else:
            max_num = result.aggregate(report_num=Max('report_num'))['report_num']
            max_count = Result.objects.filter(user_id=request.user, report_num=max_num).count()
            report_num = max_num + 1 if max_count >= 3 else max_num
        context = {
            'question': question,
            'corp_name': corp_name,
            'dept_name': dept_name,
            'next_id': next_id,
            'quest_id': quest_id,
            'quest_level': quest_level,
            'report_num': report_num
        }
        if (quest_level == '3'):
            return render(request, 'interviewapp/tendency.html', context)
        else:
            return render(request, 'interviewapp/question.html', context)



def ResultView(request):
    if request.method == 'POST':
        if request.POST['quest_level'] != '3':
            file = request.FILES['file']
            fname = file.name
            fs = FileSystemStorage(location='media/webm/')
            filename = fs.save(fname, file)
            new_fname = f"{request.user}_{request.POST['report_num']}_{request.POST['quest_id']}"
            os.system(f"ffmpeg -y -i media/webm/{filename} media/mp4/{new_fname}.mp4")
            os.system(f"ffmpeg -y -i media/webm/{filename} media/wav/{new_fname}.wav")
            os.remove(f"media/webm/{filename}")
            total, good = run_eyetrack(f'media/mp4/{new_fname}.mp4')
            feelings_faces = expression_recognition(f'media/mp4/{new_fname}.mp4')
            text, count = run_stt(f'media/wav/{new_fname}.wav', request.POST['corp_name'], request.POST['dept_name'], request.POST['quest_id'])
            user = User.objects.get(username=request.user.username)
            quest = Question.objects.get(quest_id=request.POST['quest_id'])
            Result.objects.create(user_id=user, report_num=request.POST['report_num'], quest_id=quest, result_stt=text)
        else:
            tendency = request.POST.getlist('tendency')
            str_tendency = ', '.join(tendency)
            user = User.objects.get(username=request.user.username)
            quest = Question.objects.get(quest_id=request.POST['quest_id'])
            Result.objects.create(user_id=user, report_num=request.POST['report_num'], quest_id=quest, result_add=str_tendency)

        return render(request, 'interviewapp/question.html')


def run_stt(file_path, corp_name, dept_name, quest_id):
    r = sr.Recognizer()
    harvard = sr.AudioFile(file_path)
    count = 0

    with harvard as source:
        audio = r.record(source)
        result = r.recognize_google(audio, language='ko-KR')

        if (quest_id == 1):
            talent = count_talent(result, corp_name)
            count = talent
        elif (quest_id == 2):
            job = count_job(result, dept_name)
            count = job

    return result, count


def count_talent(result, corp_name):
    naver = ['동료', '영향', '성장', '데이터', '흐름', '설득', '경험', '해석', '커뮤니케이션', '창의', '소통']
    ncsoft = ['진지', '헌신', '감동', '전문성', '도전', '창의', '열정', '실행력', '협력', '다양']

    s = result.split()  # 공백 기준으로 분리
    talent = []

    if (corp_name == '네이버'):   # 네이버
        for i in s:
            for j in naver:
                if j in i:
                    talent.append(i)
    elif (corp_name == 'NCSOFT'):   # 엔씨소프트
        for i in s:
            for j in ncsoft:
                if j in i:
                    talent.append(i)

    # print(talent)
    return len(talent)   # 인재상 포함된 단어 말한 횟수? 반환, 단어 자체를 반환하려면 talent 리스트 반환하면 됨



def count_job(result, dept_name):
    app = ['안드로이드', '아이오에스', '아이폰', '앱', '스위프트', '소켓', '배포', '플레이스토어', '자바', '코틀린', '엑스코드', '그래들']
    bigdata = ['파이썬', '빅데이터', '데이터', '판다스', '시각화', '핸들링', '전처리', '통계', '회귀', '데이터프레임', '수집']
    be = ['리액트', '스프링', '에이더블유에스', '웹', '장고', '노드제이에스', '서버', '배포', '아키텍처', '에이피아이', '프레임워크']
    qa = ['관리', '종합', '통계', '경영', '품질', '게임', '큐에이', '비용', '형상', '리스크', '검증', '테스트']
    icon = ['디자인', '아이콘', '패키지', '영상', '정보', '일러스트레이터', '포토샵', '스케치', '시각']
    text = ['자연어', '버트', '지피티', '처리', '토큰화', '모델', '트랜스포머', '분류', '벡터', '파인튜닝', '파이토치', '파이썬']

    s = result.split()  # 공백 기준으로 분리
    job = []

    # 네이버
    if (dept_name == 'iOS/Android 개발자'):
        for i in s:
            for j in app:
                if j in i:
                    job.append(i)
    elif (dept_name == '빅데이터 분석 엔지니어'):
        for i in s:
            for j in bigdata:
                if j in i:
                    job.append(i)
    elif (dept_name == 'Back-end 개발자'):
        for i in s:
            for j in be:
                if j in i:
                    job.append(i)

    # 엔씨소프트
    elif (dept_name == 'PC 온라인 게임 QA'):
        for i in s:
            for j in qa:
                if j in i:
                    job.append(i)
    elif (dept_name == '아이콘 디자이너'):
        for i in s:
            for j in icon:
                if j in i:
                    job.append(i)
    elif (dept_name == '텍스트 처리 개발자'):
        for i in s:
            for j in text:
                if j in i:
                    job.append(i)

    # print(job)
    return len(job)  # 인재상 포함된 단어 말한 횟수? 반환, 단어 자체를 반환하려면 talent 리스트 반환하면 됨



def detect_eyes(img, eye_cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None

    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]

    return left_eye, right_eye



def detect_faces(img, face_cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None

    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]

    return frame



def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img



def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)

    return keypoints



def nothing(x):
    pass



def run_eyetrack(file_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector = cv2.SimpleBlobDetector_create(detector_params)

    cap = cv2.VideoCapture(file_path)
    keypoint_list = []
    good = 0

    while True:
        ret, frame = cap.read()
        if ret:
            face_frame = detect_faces(frame, face_cascade)
            if face_frame is not None:
                eyes = detect_eyes(face_frame, eye_cascade)
                for eye in eyes:
                    if eye is not None:
                        # threshold = cv2.getTrackbarPos('threshold', 'image')
                        threshold = 120   # 전달 받은 threshold 값 넣어주기(일단 내 방 조명에 맞춰 설정한거임)
                        eye = cut_eyebrows(eye)
                        keypoints = blob_process(eye, threshold, detector)

                        if len(keypoints) != 0:
                            good += 1
                        keypoint_list.append(keypoints)

                        eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            break

    good += 20   # 사람은 1분에 평균 20회 정도 눈을 깜박인다고 함
    # print(len(keypoint_list), good)
    cap.release()

    return len(keypoint_list), good



def expression_recognition(file_path):
    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_classifier = load_model("C:/Users/minseo/mini_XCEPTION_102_66.hdf5", compile=False)   # 경로 너걸로 바꿔줘!
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    feelings_faces = []
    # for index, emotion in enumerate(EMOTIONS):
    # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

    # starting video streaming
    camera = cv2.VideoCapture(file_path)  # 녹화된 영상
    while True:
        ret, frame = camera.read()
        if ret:
            # reading the frame
            frame = imutils.resize(frame, width=300)  # your_face 창 사이즈 조절
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # your_face 창 흑백으로 바꿈
            faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frameClone = frame.copy()
            if len(faces) > 0:
                faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                feelings_faces.append(label)
            else:
                continue
        else:
            break

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # draw the label + probability bar on the canvas
            # emoji_face = feelings_faces[np.argmax(preds)]

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        #    for c in range(0, 3):
        #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
        #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
        #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

    # print(feelings_faces)
    camera.release()
    cv2.destroyAllWindows()

    return feelings_faces


