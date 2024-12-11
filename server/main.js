<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake Chat Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 600px; margin: auto; }
        .chat-box { border: 1px solid #ccc; padding: 10px; margin-top: 20px; background: #f9f9f9; }
        .dropdown { width: 100%; padding: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fake Chat Interface</h1>
        <label for="answerDropdown">Choose an Answer:</label>
        <select id="answerDropdown" class="dropdown">
            <option value="">Select an answer...</option>
        </select>
        <div class="chat-box" id="chatBox">
            <p>Select an answer to see the corresponding question.</p>
        </div>
    </div>

    <script>
        // Load JSON data
        const data = [
            { "question": "What is your name?", "answer": "My name is John." },
            { "question": "How are you?", "answer": "I'm good, thank you!" },
            { "question": "What do you do?", "answer": "I'm a software developer." }
        ];

        const answerDropdown = document.getElementById('answerDropdown');
        const chatBox = document.getElementById('chatBox');

        // Populate the dropdown with answers
        data.forEach((item, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = item.answer;
            answerDropdown.appendChild(option);
        });

        // Handle dropdown change event
        answerDropdown.addEventListener('change', () => {
            const selectedIndex = answerDropdown.value;
            if (selectedIndex) {
                const selectedPair = data[selectedIndex];
                chatBox.innerHTML = `
                    <p><strong>Answer:</strong> ${selectedPair.answer}</p>
                    <p><strong>Question:</strong> ${selectedPair.question}</p>
                `;
            } else {
                chatBox.innerHTML = `<p>Select an answer to see the corresponding question.</p>`;
            }
        });
    </script>
</body>
</html>
