import json
import time

import ollama
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

MODELS = ["gemma3:4b", "qwen2.5vl:7b"]

def ask_model(prompt, model, images=None):
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    response = ollama.chat(model=model, messages=[message], format="json")
    return json.loads(response["message"]["content"])

def detect_ingredients(image, model):
    prompt = """List the food ingredients you can see in this fridge photo.
Only include items you are reasonably sure about. Use simple names like "eggs" or "cheddar".
Respond ONLY with JSON: {"ingredients": ["", ""]}"""
    return ask_model(prompt, model, [image])["ingredients"]

def suggest_recipes(ingredients, prefs, model):
    prompt = f"""Available ingredients: {", ".join(ingredients)}.
Assume basic pantry staples (oil, salt, pepper, flour) are also available.
Diet: {prefs["diet"]}.
Try to use these first: {prefs["must_use"] or "no preference"}.
Suggest 3 dishes using mainly the available ingredients.
Respond ONLY with JSON in this format:
{{"recipes": [{{"name": "", "time": "", "ingredients": [""], "steps": [""]}}]}}"""
    return ask_model(prompt, model)["recipes"]

def compare(task, *args):
    """Run the same task on every model, timing each one."""
    results = []
    for model in MODELS:
        start = time.perf_counter()
        try:
            output, error = task(*args, model=model), None
        except Exception as e:
            output, error = [], str(e)
        results.append({
            "model": model,
            "output": output,
            "error": error,
            "seconds": round(time.perf_counter() - start, 1),
        })
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    image = request.files["photo"].read()
    results = compare(detect_ingredients, image)
    # Combine both lists, remove duplicates, keep order
    merged = list(dict.fromkeys(
        item.strip().lower() for r in results for item in r["output"]
    ))
    return render_template("confirm.html", results=results, merged=merged)

@app.route("/suggest", methods=["POST"])
def suggest():
    text = request.form.get("ingredients", "")
    ingredients = [line.strip() for line in text.splitlines() if line.strip()]
    prefs = {
        "diet": request.form.get("diet", "none"),
        "must_use": request.form.get("must_use", "").strip(),
    }
    results = compare(suggest_recipes, ingredients, prefs)
    return render_template("results.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)