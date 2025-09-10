const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const ACTIONS = {
    'ArrowLeft': 0, 'ArrowDown': 1, 'ArrowRight': 2, 'ArrowUp': 3,
    'a':0,'s':1,'d':2,'w':3
}

async function reset() {
    const res = await fetch("/api/reset");
    const data = await res.json();
    drawFrame(data.frame);
}

async function step(action) {
    const res = await fetch("/api/step", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ action })
    });
    const data = await res.json();
    drawFrame(data.frame);
}

function drawFrame(base64Img) {
    const img = new Image();
    img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = "data:image/png;base64," + base64Img;
}

document.getElementById("resetBtn").addEventListener("click", reset);
document.addEventListener("keydown", (e) => {
    if (ACTIONS.hasOwnProperty(e.key)) {
        step(ACTIONS[e.key]);
    }
});

window.onload = reset;