document.addEventListener('DOMContentLoaded', () => {
    const resultDiv = document.getElementById('result');
    
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "status_update") {
            resultDiv.style.display = 'flex'; 
            if (request.prediction === 1) {
                resultDiv.textContent = `SENSITIVE (Conf: ${(request.confidence * 100).toFixed(0)}%)`;
                resultDiv.className = 'status-box inactive'; 
            } else {
                resultDiv.textContent = `SAFE (Conf: ${(request.confidence * 100).toFixed(0)}%)`;
                resultDiv.className = 'status-box active'; 
            }
        }
    });
});

document.getElementById('checkBtn').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: () => { 
            chrome.tabs.sendMessage(tab.id, { action: "manual_check" });
        } 
    });
    window.close();
});
