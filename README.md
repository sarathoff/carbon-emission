# Carbon Audit

This application analyzes energy bills and other documents to calculate carbon emissions.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up the Gemini API key:**

    - Create a file named `.env` in the root of the project.
    - Add the following line to the `.env` file, replacing `your_api_key_here` with your actual Gemini API key:

      ```
      GEMINI_API_KEY=your_api_key_here
      ```

3.  **Run the application:**

    ```bash
    streamlit run unified.py
    ```
