const video = document.querySelector("#video");

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then (function (stream) {
            video.srcObject = stream;
            video.play()
        })
        .catch (function (error) {
            console.log("ERROR");
        });
}