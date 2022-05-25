const video = document.querySelector("#video");
const slide = document.getElementById("myRange");
const y = document.getElementById("thres-value");

y.innerHTML = slide.value;

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

slide.oninput = function() {
    y.innerHTML = this.value;
}