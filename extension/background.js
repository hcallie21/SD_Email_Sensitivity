// Background script to handle requests and avoid Mixed Content issues

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "check_sensitivity") {
    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: request.text })
    })
    .then(response => response.json())
    .then(data => sendResponse({ success: true, data: data }))
    .catch(error => sendResponse({ success: false, error: error.message }));
    
    return true; 
  }
});
