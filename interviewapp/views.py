import json

from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.db.models import Max, Count
from django.shortcuts import render
from interviewapp.models import Question, Result

import os
import speech_recognition as sr


# Create your views here.
def QuestionView(request):
    if request.method == 'GET':
        corp_name = request.GET.get('corp', None)
        dept_name = request.GET.get('dept', None)
        quest_num = request.GET.get('question', None)
        next_num = str(int(quest_num) + 1)
        question = Question.objects.filter(quest_id=quest_num)
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
            'next_num': next_num,
            'quest_id': quest_num,
            'report_num': report_num
        }
        if (quest_num == '4') or (quest_num == '5'):
            return render(request, 'interviewapp/tendency.html', context)
        else:
            return render(request, 'interviewapp/question.html', context)


def TestView(request):
    if request.method == 'GET':
        video = request.GET.get('video', None);
        context = {'video': video}
        return render(request, 'interviewapp/test.html', context)
    else:
        file = request.FILES['file']
        fname = file.name
        if file.name[-4:] == 'webm':
            fs = FileSystemStorage(location='media/webm/')
            filename = fs.save(fname, file)
            new_fname = f"{request.user}_{request.POST['quest_id']}"
            os.system(f"ffmpeg -y -i media/webm/{filename} media/mp4/{new_fname}.mp4")
            os.system(f"ffmpeg -y -i media/webm/{filename} media/wav/{new_fname}.wav")
            text = run_stt(f'media/wav/{new_fname}.wav')
            user = User.objects.get(username=request.user.username)
            quest = Question.objects.get(quest_id=request.POST['quest_id'])
            Result.objects.create(user_id=user, report_num=request.POST['report_num'], quest_id=quest, result_stt=text)
        else:
            fs = FileSystemStorage(location='media')
            filename = fs.save(fname, file)
        return render(request, 'interviewapp/question.html')


# def TendencyView(request):



def run_stt(file_path):
    r = sr.Recognizer()
    harvard = sr.AudioFile(file_path)
    with harvard as source:
        audio = r.record(source)
        result = r.recognize_google(audio, language='ko-KR')
    return result