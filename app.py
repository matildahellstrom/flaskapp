import json

import ollama
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

MODEL = "gemma3:4b"

def ask_model(prompt, images=None):
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    response = ollama.chat(model=MODEL, messages=[message], format="json")
    return json.loads(response["message"]["content"])

def detect_ingredients(photo):
    prompt = """List the food ingredients you can see in this fridge photo.
Only include items you are reasonably sure about. Use simple names like "eggs" or "cheddar".
Respond ONLY with JSON: {"ingredients": ["", ""]}"""
    try:
        return ask_model(prompt, [photo.read()])["ingredients"]
    except (json.JSONDecodeError, KeyError):
        return []

def suggest_recipes(ingredients, prefs):
    prompt = f"""Available ingredients: {", ".join(ingredients)}.
Assume basic pantry staples (oil, salt, pepper, flour) are also available.
Diet: {prefs["diet"]}.
Try to use these first: {prefs["must_use"] or "no preference"}.
Suggest 3 dishes using mainly the available ingredients.
Respond ONLY with JSON in this format:
{{"recipes": [{{"name": "", "time": "", "ingredients": [""], "steps": [""]}}]}}"""
    try:
        return ask_model(prompt)["recipes"]
    except (json.JSONDecodeError, KeyError):
        return [{"name": "Couldn't parse a response", "time": "",
                 "ingredients": [], "steps": ["Try again"]}]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    photo = request.files["photo"]
    ingredients = detect_ingredients(photo)
    return render_template("confirm.html", ingredients=ingredients)

@app.route("/suggest", methods=["POST"])
def suggest():
    text = request.form.get("ingredients", "")
    ingredients = [line.strip() for line in text.splitlines() if line.strip()]
    prefs = {
        "diet": request.form.get("diet", "none"),
        "must_use": request.form.get("must_use", "").strip(),
    }
    recipes = suggest_recipes(ingredients, prefs)
    return render_template("results.html", recipes=recipes)

if __name__ == "__main__":
    app.run(debug=True)