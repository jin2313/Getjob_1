const $video_realtime = document.querySelector("#video-realtime");
const $btn_start = document.querySelector("#video-start");
const $btn_stop = document.querySelector("#video-stop");
const download = document.getElementById('download');

let mediaRecorder = null; // MediaRecorder(녹화기) 변수 선언
const arrVideoData = []; // 영상 데이터를 담아줄 배열 선언

// 시작 버튼 이벤트 처리
$btn_start.onclick = async function(event) {
    // 카메라 입력영상 스트림 생성
    const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: true
    });

    // 실시간 영상 재생 처리: 첫번째 video태그에서 재생
    $video_realtime.srcObject = mediaStream;
    $video_realtime.onloadedmetadata = (event)=> {
        $video_realtime.play();
    }

    // mediaRecorder객체(녹화기) 생성
    mediaRecorder = new MediaRecorder(mediaStream);

    // 녹화 데이터 입력 이벤트 처리
    mediaRecorder.ondataavailable = (event)=> {
        arrVideoData.push(event.data); // 녹화 데이터(Blob)가 들어올 때마다 배열에 담아두기
    }

    // 녹화 종료 이벤트 처리
    mediaRecorder.onstop = (event)=> {
        const videoBlob = new Blob(arrVideoData); // 배열에 담아둔 녹화 데이터들을 통합한 Blob객체 생성
        const blobURL = window.URL.createObjectURL(videoBlob); // BlobURL(ObjectURL) 생성

        download.href = blobURL;
        download.download = 'test.mp4'
        $("#download").get(0).click();

        // 기존 녹화 데이터 제거
        arrVideoData.splice(0);
    }

    // 녹화 시작!
    mediaRecorder.start();
}

// 종료 버튼 이벤트 처리
$btn_stop.onclick = (event)=> {
    // 녹화 종료!
    mediaRecorder.stop();
}
