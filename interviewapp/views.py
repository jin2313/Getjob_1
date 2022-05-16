from django.shortcuts import render
from interviewapp.models import Question


# Create your views here.
def QuestionView(request):
    if request.method == 'GET':
        corp_name = request.GET.get('corp', None)
        dept_name = request.GET.get('dept', None)
        quest_num = request.GET.get('question', None)
        question = Question.objects.filter(quest_id=quest_num)
        print(question)
        context = {
            'question': question,
            'corp_name': corp_name,
            'dept_name': dept_name,
            'quest_num': quest_num
        }
        return render(request, 'interviewapp/question.html', context)
    else:
        user = request.user
        print(user)
        # user로 결과 보고서 검색, count해서 +1 한 값을 결과 보고서 id로
        return render()


def WebcamView(request):
    return render(request, 'interviewapp/webcam.html')