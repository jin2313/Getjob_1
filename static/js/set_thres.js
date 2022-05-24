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


var slide = document.getElementById("myRange");
var y = document.getElementById("f");
y.innerHTML = slide.value;

slide.oninput = function() {
    y.innerHTML = this.value;
}
