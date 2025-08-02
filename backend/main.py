from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List
import uvicorn
import asyncio
import openai
import os
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json
from dotenv import load_dotenv


app = FastAPI(
    title="Travel Recommendation API",
    description="AI-powered travel recommendation system with ML satisfaction prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class UserPreferences(BaseModel):
    climate: str = Field(..., regex="^(tropical|moderate|cold)$")
    duration: int = Field(..., ge=1, le=14)
    budget: int = Field(..., ge=500, le=5000)
    interests: List[str] = Field(..., min_items=1)

    @validator('interests')
    def validate_interests(cls, v):
        valid_interests = {'adventure', 'culture', 'relaxation', 'food'}
        for interest in v:
            if interest not in valid_interests:
                raise ValueError(f'Invalid interest: {interest}')
        return v

class Destination(BaseModel):
    name: str
    description: str
    reasoning: str
    satisfactionScore: int
    estimatedCost: int
    climate: str
    bestDuration: str

class RecommendationResponse(BaseModel):
    destinations: List[Destination]
    message: str
    preferences: UserPreferences
    timestamp: str

class SatisfactionPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.climate_encoder = LabelEncoder()
        self.is_trained = False
        self._train_model()

    def _generate_training_data(self):
        np.random.seed(42)
        n_samples = 100
        climates = np.random.choice(['tropical', 'moderate', 'cold'], n_samples)
        durations = np.random.randint(1, 15, n_samples)
        budgets = np.random.randint(500, 5001, n_samples)
        dest_costs = np.random.randint(800, 4000, n_samples)
        interest_matches = np.random.randint(0, 5, n_samples)
        climate_matches = np.random.choice([0, 1], n_samples)

        satisfaction_scores = []
        for i in range(n_samples):
            score = 50
            score += climate_matches[i] * 25
            score += (interest_matches[i] / 4) * 30
            budget_ratio = min(budgets[i] / dest_costs[i], 2)
            score += min(budget_ratio * 15, 25)
            if 7 <= durations[i] <= 10:
                score += 10
            else:
                score += max(0, 10 - abs(durations[i] - 8.5))
            score += np.random.normal(0, 5)
            score = max(0, min(100, score))
            satisfaction_scores.append(score)

        return {
            'climates': climates,
            'durations': durations,
            'budgets': budgets,
            'dest_costs': dest_costs,
            'interest_matches': interest_matches,
            'climate_matches': climate_matches,
            'satisfaction_scores': satisfaction_scores
        }

    def _train_model(self):
        data = self._generate_training_data()
        climate_encoded = self.climate_encoder.fit_transform(data['climates'])
        X = np.column_stack([
            climate_encoded,
            data['durations'],
            data['budgets'],
            data['dest_costs'],
            data['interest_matches'],
            data['climate_matches']
        ])
        y = np.array(data['satisfaction_scores'])
        self.model.fit(X, y)
        self.is_trained = True

    def predict_satisfaction(self, climate: str, duration: int, budget: int,
                             dest_cost: int, interest_matches: int, climate_match: bool):
        if not self.is_trained:
            return 75
        try:
            climate_encoded = self.climate_encoder.transform([climate])[0]
            features = np.array([[climate_encoded, duration, budget, dest_cost, interest_matches, int(climate_match)]])
            score = self.model.predict(features)[0]
            return max(0, min(100, int(round(score))))
        except Exception:
            return 75

satisfaction_predictor = SatisfactionPredictor()

DESTINATIONS = [
    {"name": "Bali, Indonesia", "climate": "tropical", "base_cost": 1200,
     "interests": ["relaxation", "culture", "food"], "best_duration": "7-10 days"},
    {"name": "Swiss Alps, Switzerland", "climate": "cold", "base_cost": 3000,
     "interests": ["adventure", "relaxation"], "best_duration": "5-7 days"},
    {"name": "Tokyo, Japan", "climate": "moderate", "base_cost": 2500,
     "interests": ["culture", "food", "adventure"], "best_duration": "8-12 days"},
    {"name": "Costa Rica", "climate": "tropical", "base_cost": 1800,
     "interests": ["adventure", "relaxation"], "best_duration": "9-12 days"},
    {"name": "Tuscany, Italy", "climate": "moderate", "base_cost": 2200,
     "interests": ["culture", "food", "relaxation"], "best_duration": "7-10 days"},
    {"name": "Iceland", "climate": "cold", "base_cost": 2800,
     "interests": ["adventure", "relaxation"], "best_duration": "6-8 days"},
    {"name": "Thailand", "climate": "tropical", "base_cost": 1000,
     "interests": ["culture", "food", "relaxation", "adventure"], "best_duration": "10-14 days"},
    {"name": "Morocco", "climate": "moderate", "base_cost": 1500,
     "interests": ["culture", "adventure", "food"], "best_duration": "8-10 days"}
]

async def get_ai_recommendations(preferences: UserPreferences) -> List[dict]:
    system_msg = {"role": "system", "content": "You are a helpful travel recommendation expert."}

    user_prompt = {
        "role": "user",
        "content": (
            f"You are a travel recommendation AI. Based on the preferences below, "
            f"select 3-5 best matching destinations from this list:\n\n"
            f"{[d['name'] for d in DESTINATIONS]}\n\n"
            f"User Preferences:\n"
            f"- Climate: {preferences.climate}\n"
            f"- Duration: {preferences.duration} days\n"
            f"- Budget: ${preferences.budget:,}\n"
            f"- Interests: {', '.join(preferences.interests)}\n\n"
            f"Available destinations with details:\n{json.dumps(DESTINATIONS, indent=2)}\n\n"
            f"Respond ONLY in this JSON format:\n"
            f"[\n"
            f"  {{\n"
            f"    \"name\": string,\n"
            f"    \"description\": string,\n"
            f"    \"reasoning\": string\n"
            f"  }}, ... up to 5 items\n"
            f"]"
        )
    }

    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[system_msg, user_prompt],
            temperature=0.7,
            max_tokens=1000
        )

        ai_response = response.choices[0].message.content

        # Log the raw response for debugging
        print("🧠 AI raw response:\n", ai_response)

        try:
            recommendations = json.loads(ai_response)
            return recommendations
        except json.JSONDecodeError as je:
            print("❌ Failed to parse AI JSON:", je)
            return get_fallback_recommendations(preferences)

    except Exception as e:
        print("❌ OpenAI call failed:", e)
        return get_fallback_recommendations(preferences)

def get_fallback_recommendations(preferences: UserPreferences) -> List[dict]:
    scored_destinations = []
    for dest in DESTINATIONS:
        score = 0
        if dest["climate"] == preferences.climate:
            score += 40
        matching_interests = len(set(dest["interests"]) & set(preferences.interests))
        score += matching_interests * 15
        if dest["base_cost"] <= preferences.budget:
            score += 30
        elif dest["base_cost"] <= preferences.budget * 1.2:
            score += 15
        duration_range = dest["best_duration"].replace(" days", "").split("-")
        min_days = int(duration_range[0].strip())
        max_days = int(duration_range[1].strip()) if len(duration_range) > 1 else min_days

        if min_days <= preferences.duration <= max_days:
            score += 15
        scored_destinations.append((dest, score))
    scored_destinations.sort(key=lambda x: x[1], reverse=True)
    top_destinations = scored_destinations[:5]
    recommendations = []
    for dest, _ in top_destinations:
        description = f"{dest['name']} offers an incredible {dest['climate']} climate experience. Perfect for travelers seeking {' and '.join(dest['interests'][:2])} adventures."
        reasoning_parts = []
        if dest["climate"] == preferences.climate:
            reasoning_parts.append(f"matches your {preferences.climate} climate preference")
        matching_interests = set(dest["interests"]) & set(preferences.interests)
        if matching_interests:
            reasoning_parts.append(f"aligns with your interest in {', '.join(matching_interests)}")
        if dest["base_cost"] <= preferences.budget:
            reasoning_parts.append(f"fits well within your ${preferences.budget:,} budget")
        reasoning = f"This destination {', and '.join(reasoning_parts)}."
        recommendations.append({
            "name": dest["name"],
            "description": description,
            "reasoning": reasoning
        })
    return recommendations

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(preferences: UserPreferences):
    try:
        ai_recommendations = await get_ai_recommendations(preferences)
        processed_destinations = []
        for ai_rec in ai_recommendations:
            dest_data = next((d for d in DESTINATIONS if d["name"] == ai_rec["name"]), None)
            if dest_data:
                matching_interests = len(set(dest_data["interests"]) & set(preferences.interests))
                climate_match = dest_data["climate"] == preferences.climate
                satisfaction_score = satisfaction_predictor.predict_satisfaction(
                    climate=preferences.climate,
                    duration=preferences.duration,
                    budget=preferences.budget,
                    dest_cost=dest_data["base_cost"],
                    interest_matches=matching_interests,
                    climate_match=climate_match
                )
                processed_destinations.append(Destination(
                    name=ai_rec["name"],
                    description=ai_rec["description"],
                    reasoning=ai_rec["reasoning"],
                    satisfactionScore=satisfaction_score,
                    estimatedCost=dest_data["base_cost"],
                    climate=dest_data["climate"],
                    bestDuration=dest_data["best_duration"]
                ))
        processed_destinations.sort(key=lambda x: x.satisfactionScore, reverse=True)
        await asyncio.sleep(1.5)
        return RecommendationResponse(
            destinations=processed_destinations[:5],
            message="AI recommendations generated successfully",
            preferences=preferences,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_model_trained": satisfaction_predictor.is_trained
    }

@app.get("/")
async def root():
    return {
        "message": "Travel Recommendation API with AI Integration",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3001,
        reload=True,
        log_level="info"
    )

