import json

import ollama
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

MODEL = "gemma3:4b"

def get_recipes(photo, prefs):
    prompt = f"""Look at this photo of a fridge. List the ingredients you can see,
then suggest 3 dishes using mainly those, assuming basic pantry staples.
Diet: {prefs['diet']}.
Respond ONLY with JSON in this format:
{{"recipes": [{{"name": "", "time": "", "ingredients": [""], "steps": [""]}}]}}"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt,
                   "images": [photo.read()]}],
        format="json",
    )
    try:
        return json.loads(response["message"]["content"])["recipes"]
    except (json.JSONDecodeError, KeyError):
        return [{"name": "Couldn't parse a response", "time": "",
                 "ingredients": [], "steps": ["Try again or another photo"]}]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/suggest", methods=["POST"])
def suggest():
    photo = request.files["photo"]
    prefs = {"diet": request.form.get("diet", "none")}
    recipes = get_recipes(photo, prefs)
    return render_template("results.html", recipes=recipes, filename=photo.filename)

if __name__ == "__main__":
    app.run(debug=True)