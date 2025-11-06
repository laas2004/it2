// Simple message passing between iframes
function startCall() {
    const videoFrame = document.getElementById('videoFrame');
    videoFrame.contentWindow.postMessage({ action: 'startCall' }, '*');
}

function shareText() {
    const text2signFrame = document.getElementById('text2signFrame');
    const sign2textFrame = document.getElementById('sign2textFrame');
    
    // Get text from text-to-sign app and send to sign-to-text app
    text2signFrame.contentWindow.postMessage({ action: 'getTranslation' }, '*');
}

function shareSign() {
    const sign2textFrame = document.getElementById('sign2textFrame');
    const text2signFrame = document.getElementById('text2signFrame');
    
    // Get sign from sign-to-text app and send to text-to-sign app
    sign2textFrame.contentWindow.postMessage({ action: 'getTranslation' }, '*');
}

// Listen for messages from iframes
window.addEventListener('message', function(event) {
    // Handle communication between different components
    console.log('Message received:', event.data);
});