const analyzeButton = document.getElementById("analyzeButton");

analyzeButton.addEventListener("click", function () {

    const fileInput = document.getElementById("transcriptFile");
    const results = document.getElementById("results");

    if (fileInput.files.length === 0) {
        results.textContent = "Please choose a transcript first.";
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    // Show loading indicator while heavy processing runs
    results.textContent = "Analyzing transcript and generating citations...";

    reader.onload = async function(event) {

        const transcript = event.target.result;

        try {
            // Send the transcript to FastAPI (matching our port, e.g., 8081)
            const response = await fetch("http://127.0.0.1:8081/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    transcript: transcript
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }

            // Convert the response into JSON
            const data = await response.json();

            // Clear previous results
            results.innerHTML = "";

            // Inject the rendered HTML returned from BeautifulSoup directly
            if (data.html) {
                results.innerHTML = data.html;
            } else {
                results.textContent = "No HTML content was generated.";
            }

        } catch (error) {
            console.error("Error analyzing transcript:", error);
            results.textContent = "An error occurred while fetching analysis results.";
        }

    };

    reader.readAsText(file);

});
