
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const downloadPdfButton = document.getElementById('download-pdf-button');
    const chatSection = document.getElementById('chat-section');
    const chatHistory = document.getElementById('chat-history');
    const chatMessage = document.getElementById('chat-message');
    const sendButton = document.getElementById('send-button');
    const loadSamplesButton = document.getElementById('load-samples-button');
    const sampleImagesContainer = document.getElementById('sample-images-container');

    let extractionResults = [];

    loadSamplesButton.addEventListener('click', async () => {
        const response = await fetch('/sample_images');
        const images = await response.json();
        sampleImagesContainer.innerHTML = '';
        images.forEach(image => {
            const imgElement = document.createElement('img');
            imgElement.src = `/data/raw_images/${image}`;
            imgElement.alt = image;
            imgElement.classList.add('sample-image');
            imgElement.addEventListener('click', () => {
                const dataTransfer = new DataTransfer();
                fetch(imgElement.src)
                    .then(res => res.blob())
                    .then(blob => {
                        const file = new File([blob], image, { type: blob.type });
                        dataTransfer.items.add(file);
                        fileInput.files = dataTransfer.files;
                        uploadButton.click();
                    });
            });
            sampleImagesContainer.appendChild(imgElement);
        });
    });

    uploadButton.addEventListener('click', async () => {
        const files = fileInput.files;
        if (files.length === 0) {
            alert('Please select files to upload.');
            return;
        }

        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        extractionResults = await response.json();
        displayResults(extractionResults);
        resultsSection.classList.remove('hidden');
        chatSection.classList.remove('hidden');
    });

    function displayResults(results) {
        resultsContainer.innerHTML = '';
        results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.innerHTML = `<h3>${result.filename}</h3>`;
            const table = document.createElement('table');
            table.innerHTML = `
                <tr>
                    <th>Energy Type</th>
                    <th>Consumption</th>
                    <th>Emissions (kg CO2e)</th>
                </tr>
            `;
            for (const [energyType, data] of Object.entries(result.extracted_data)) {
                const row = table.insertRow();
                row.innerHTML = `
                    <td>${energyType.replace('_', ' ').title()}</td>
                    <td>${data.amount} ${data.unit}</td>
                    <td>${data.emissions_kg_co2e.toFixed(2)}</td>
                `;
            }
            resultDiv.appendChild(table);
            resultsContainer.appendChild(resultDiv);
        });
    }

    downloadPdfButton.addEventListener('click', async () => {
        const response = await fetch('/download_pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(extractionResults)
        });

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'emissions_report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    });

    sendButton.addEventListener('click', async () => {
        const message = chatMessage.value;
        if (!message) return;

        appendMessage('user', message);
        chatMessage.value = '';

        const context = JSON.stringify(extractionResults);
        const response = await fetch('/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: context })
        });

        const data = await response.json();
        appendMessage('assistant', data.summary);
    });

    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', sender);
        messageDiv.innerText = message;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
