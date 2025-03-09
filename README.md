# FashInsta - AI Fashion Assistant

FashInsta is an AI-powered fashion assistant that helps users create personalized outfit recommendations based on their style preferences, occasion, and budget.

## Features

- Upload and analyze personal style photos
- Get outfit recommendations based on occasion and preferences
- Customizable color and style preferences
- Budget-aware outfit suggestions
- Interactive web interface

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FashInsta.git
cd FashInsta
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
FashInsta/
├── app.py                    # Streamlit web application
├── api_server.py            # FastAPI backend server
├── wardrobe_recommender.py  # Core recommendation engine
├── outfit_visualizer.py     # Outfit visualization module
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Running the Application

1. Start the API server:
```bash
python api_server.py
```
The API server will run on http://localhost:8504

2. In a new terminal, start the Streamlit app:
```bash
streamlit run app.py
```
The web interface will be available at http://localhost:8501

## Usage

1. Open the web interface in your browser
2. (Optional) Upload photos of your typical style
3. Select your preferences:
   - Occasion (Business Meeting, Party, etc.)
   - Budget
   - Color preferences
   - Style preferences
4. Click "Get Recommendations" to receive personalized outfit suggestions

## Dependencies

- streamlit
- fastapi
- pillow
- torch
- transformers
- pandas
- scikit-learn
- python-multipart
- uvicorn
- requests

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 