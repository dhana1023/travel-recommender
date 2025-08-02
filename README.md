# Travel Recommendation System

This project is a simple AI-powered travel recommendation system built with FastAPI (Python) and a minimal HTML frontend.

##  Features

-  Suggests 3–5 travel destinations based on:
  - Preferred climate
  - Travel duration
  - Budget
  - Interests (adventure, culture, food, relaxation)
-  Uses OpenAI GPT (gpt-3.5-turbo) for destination recommendation generation
-  Predicts user satisfaction score (0–100) using a simple RandomForestRegressor model
-  Has fallback logic in case of LLM failure
-  Clean UI with basic CSS

##  Project Structure

```
travel-recommender/
├── README.md
├── frontend/
│   └── index.html
├── backend/
│   └── main.py
├── requirements.txt
└── sample_output.json
```


### Backend Setup

1. **Clone the repo**
```bash
git clone https://github.com/dhana1023/travel-recommender.git
cd travel-recommender/backend
```

2. **Create a virtual environment & install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r ../requirements.txt
```

3. **Set your OpenAI API key**
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

4. **Run the FastAPI server**
```bash
uvicorn main:app --reload --port 3001
```

### Frontend Setup

Open the `frontend/index.html` in your browser.

> Make sure the backend is running at http://localhost:3001

##  Satisfaction Prediction

The satisfaction score is generated based on features like:
- Budget vs destination cost
- Matching interests count
- Climate alignment
- Duration match

It uses a `RandomForestRegressor` trained on synthetic data.

##  Sample API Request

POST `/api/recommendations`

```json
{
  "climate": "tropical",
  "duration": 7,
  "budget": 2000,
  "interests": ["adventure", "food"]
}
```

##  Sample Output (stored in `sample_output.json`)

```json
{
  "destinations": [
    {
      "name": "Bali, Indonesia",
      "description": "...",
      "reasoning": "...",
      "satisfactionScore": 87,
      "estimatedCost": 1200,
      "climate": "tropical",
      "bestDuration": "7-10 days"
    },
    ...
  ],
  "message": "AI recommendations generated successfully",
  "timestamp": "..."
}
```

##  Health Check

Check server status at:
```
GET /api/health
```

##  Assumptions

- GPT-3.5 is used (can be swapped with other OpenAI-compatible models)
- Basic frontend; no login/authentication required
- No persistent database (everything runs in-memory)

---

© 2025 Dhananjani Jayarukshi. All rights reserved.
