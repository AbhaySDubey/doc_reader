
#### Find live web-app at: https://huggingface.co/spaces/intuitive262/Doc_Reader


# Document Reader using OCR (English/Hindi)

This project is a document reader that uses Optical Character Recognition (OCR) to extract text from images. It supports both English and Hindi languages. The application allows users to upload an image, extract text from it, and search within the extracted text.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.9 or later.
- You have a Windows, macOS, or Linux machine.
- You have installed the following Python libraries:
  - `byaldi`
  - `transformers`
  - `torch`
  - `qwen_vl_utils`
  - `Pillow`
  - `gradio`
- You can install necessary dependencies, by running 
    '''
        pip install -r requirements.txt
    '''

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AbhaySDubey/doc_reader.git
    cd doc_reader
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:7860`.

3. Upload an image using the "Upload an Image" button.

4. Enter a query in the "Enter query for retrieval" textbox.

5. Click the "Read Doc!" button to extract text from the image.

6. Enter a search term in the "Enter search term" textbox and click the "Search!" button to search within the extracted text.

7. Click the "Download Plain Text" button to download the extracted text as a plain text file.

## Example

1. Upload an image containing text.
2. Use the prompt: "What text can you identify in this image? Include everything, even if it's partially obscured or in the background."
3. Extract the text and search for specific terms within the extracted text.
