import json
from datetime import time

from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.db.models import Max, Count
from django.shortcuts import render
from interviewapp.models import Question, Result

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
            text = run_stt(f'media/wav/{new_fname}.wav')
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


def SettingView(request):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector = cv2.SimpleBlobDetector_create(detector_params)

    cap = cv2.VideoCapture('http://127.0.0.1:8000/thres/')  # 웹캠 사용(아직 안됨)
    # cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    # threshold = cv2.getTrackbarPos('threshold', 'image')
                    threshold = 112
                    eye = cut_eyebrows(eye)
                    # keypoints = blob_process(eye, threshold, detector)
                    keypoints = blob_process(eye, threshold, detector)
                    print(keypoints)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    return render(request, 'interviewapp/threshold.html', {'cap': cap, })




def run_stt(file_path):
    r = sr.Recognizer()
    harvard = sr.AudioFile(file_path)
    with harvard as source:
        audio = r.record(source)
        result = r.recognize_google(audio, language='ko-KR')
    return result



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
