let countInterval

function Timer(duration, condition) {   // 1초씩 카운트
    var timer = duration, minutes, seconds;
    var timer_text = document.getElementById("timer-text");
    var stop_btn = document.getElementById("video-stop");

    countInterval = setInterval(function () {
        minutes = parseInt(timer / 60, 10);
        seconds = parseInt(timer % 60, 10);
        minutes = minutes < 10 ? "0" + minutes : minutes;
        seconds = seconds < 10 ? "0" + seconds : seconds;

        timer_text.innerText = minutes + ":" + seconds;
        timer--;

        if (timer < 0) {
            clearInterval(countInterval);

            if (condition == "initial") {
                stop_btn.disabled = false;
                stop_btn.style.backgroundColor = "#8ccac3";
            }
        }
    }, 1000);
}
window.onload = function() {
    Timer(10, "initial");
};

function Start() {

    clearInterval(countInterval);
}