/*
camera.js
---------
Adds live camera capture to the existing Legal Metrology Compliance Checker.
Does NOT modify the /check endpoint - captured frames are converted to a
JPEG Blob and sent through the exact same FormData upload path as a file
picker selection, so the existing backend needs zero changes.

Include with: <script src="camera.js"></script>
Requires these elements to exist in index.html (see camera_ui_snippet.html):
  #cameraVideo, #cameraCanvas, #startCameraBtn, #captureBtn,
  #retakeBtn, #cameraPreview, #cameraCheckBtn
*/

let cameraStream = null;
let capturedBlob = null;

async function startCamera() {
  const video = document.getElementById('cameraVideo');
  const startBtn = document.getElementById('startCameraBtn');
  try {
    // 'environment' prefers the rear camera on phones; falls back on laptops
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });
    video.srcObject = cameraStream;
    video.style.display = 'block';
    startBtn.style.display = 'none';
    document.getElementById('captureBtn').style.display = 'inline-block';
  } catch (err) {
    alert(
      'Could not access camera. Check browser permissions, and note ' +
      'camera access requires HTTPS or localhost (http://127.0.0.1 is fine).'
    );
    console.error('Camera error:', err);
  }
}

function captureFrame() {
  const video = document.getElementById('cameraVideo');
  const canvas = document.getElementById('cameraCanvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    capturedBlob = blob;
    const previewImg = document.getElementById('cameraPreview');
    previewImg.src = URL.createObjectURL(blob);
    previewImg.style.display = 'block';

    // Swap to review mode
    video.style.display = 'none';
    document.getElementById('captureBtn').style.display = 'none';
    document.getElementById('retakeBtn').style.display = 'inline-block';
    document.getElementById('cameraCheckBtn').style.display = 'inline-block';
  }, 'image/jpeg', 0.92);
}

function retakePhoto() {
  const video = document.getElementById('cameraVideo');
  video.style.display = 'block';
  document.getElementById('cameraPreview').style.display = 'none';
  document.getElementById('captureBtn').style.display = 'inline-block';
  document.getElementById('retakeBtn').style.display = 'none';
  document.getElementById('cameraCheckBtn').style.display = 'none';
  capturedBlob = null;
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
}

// Sends the captured frame through the SAME /check endpoint used by the
// existing file-upload flow. renderResult() and loadHistory() are the
// functions already defined in index.html - reused as-is.
async function checkCameraCapture() {
  if (!capturedBlob) {
    alert('Capture a photo first');
    return;
  }
  const btn = document.getElementById('cameraCheckBtn');
  btn.disabled = true;
  btn.textContent = 'Checking...';

  const formData = new FormData();
  formData.append('file', capturedBlob, 'camera_capture.jpg');

  // Feature: Geotagged Inspections - best-effort, never blocks the scan
  const loc = await getLocationIfRequested('geotagCamera');
  if (loc) {
    formData.append('latitude', loc.latitude);
    formData.append('longitude', loc.longitude);
  }

  try {
    const res = await fetch('/check', { method: 'POST', body: formData });
    const data = await res.json();
    renderResult(data);   // existing function in index.html - untouched
    loadHistory();        // existing function in index.html - untouched
    stopCamera();
  } catch (e) {
    document.getElementById('result').innerHTML =
      '<p style="color:red">Error checking camera image. Is the server running?</p>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Check compliance';
  }
}
