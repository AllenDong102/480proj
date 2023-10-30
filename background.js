chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
    if (changeInfo.status == 'complete' && tab.active) {
        var url = tab.url;
        sendDataToServer(url);
    }
});

function sendDataToServer(url) {
    fetch('http://127.0.0.1:5000/track', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({url: url})
    });
}

