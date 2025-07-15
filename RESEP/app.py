from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
import ast

# === FLASK APP ===
app = Flask(__name__)

# === LOAD & PARSE DATASET ===
df = pd.read_csv('reseptrainingg.csv')
print("CSV Loaded:", df.shape)

# Fungsi membersihkan Ingredients dari CSV
def clean_ingredient_text(s):
    try:
        if isinstance(s, str):
            # Coba parsing sebagai Python literal dulu
            try:
                return ast.literal_eval(s)
            except:
                s = s.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1]
                ingredients = []
                for item in s.split(','):
                    item = item.strip().strip('"\'')
                    if item:
                        ingredients.append(item)
                return ingredients
        return []
    except Exception as e:
        print(f"Parsing error: {e}")
        return []

df['Ingredients'] = df['Ingredients'].apply(clean_ingredient_text)

# Parsing kuantitas bahan
def parse_ingredients(ingredient_list):
    ingredient_dict = {}
    if not isinstance(ingredient_list, list) or not ingredient_list:
        return ingredient_dict
    
    for item in ingredient_list:
        if not isinstance(item, str):
            continue
            
        item = item.strip().lower()
        pattern = r"([0-9./]+)?\s*(kg|gram|gr|g|sendok teh|sdt|sdm|butir|biji|lembar|potong|siung|buah|cup|ml|liter|l|)\s*(.*)"
        match = re.match(pattern, item)
        
        if match:
            qty_str, unit, name = match.groups()
            qty = 1.0
            if qty_str:
                try:
                    if '/' in qty_str:
                        a, b = map(float, qty_str.split('/'))
                        qty = a / b
                    else:
                        qty = float(qty_str.replace(',', '.'))
                except:
                    pass

            konversi = {
                'kg': 1000,
                'gram': 1,
                'gr': 1,
                'g': 1,
                'sendok teh': 5,
                'sdt': 5,
                'sdm': 15,
                'butir': 50,
                'biji': 50,
                'siung': 3,
                'lembar': 10,
                'potong': 100,
                'buah': 100,
                'cup': 240,
                'ml': 1,
                'liter': 1000,
                'l': 1000
            }
            multiplier = konversi.get(unit.strip(), 1)
            total_qty = qty * multiplier

            name = name.strip()
            if name:
                main_name = re.search(r'\b(\w+)\b', name)
                if main_name:
                    name = main_name.group(1)
                ingredient_dict[name] = total_qty
        else:
            main_name = re.search(r'\b(\w+)\b', item)
            if main_name:
                name = main_name.group(1)
                ingredient_dict[name] = 100  # fallback 100g
    
    return ingredient_dict

df['Parsed'] = df['Ingredients'].apply(parse_ingredients)

# Ambil semua nama bahan sebagai kolom vektor
all_ingredients = set()
for parsed in df['Parsed']:
    all_ingredients.update(parsed.keys())

ingredient_columns = list(all_ingredients)
print(f"Total unique ingredients: {len(ingredient_columns)}")
if len(ingredient_columns) > 0:
    print(f"Sample ingredients: {ingredient_columns[:5]}")

def ingredient_to_vector(parsed):
    return [parsed.get(i, 0) for i in ingredient_columns]

X_ingredients = np.array([ingredient_to_vector(p) for p in df['Parsed']])

# Latih model KNN
knn = None
if len(ingredient_columns) > 0:
    knn = NearestNeighbors(n_neighbors=min(5, len(df)), metric='euclidean')
    knn.fit(X_ingredients)
    print("KNN model trained successfully")
else:
    print("ERROR: No ingredients found to train model")

# Hitung porsi maksimal
def calculate_portions(user_stock, recipe_stock):
    porsis = []
    for bahan, qty in recipe_stock.items():
        if bahan in user_stock and qty > 0:
            porsis.append(user_stock[bahan] / qty)
        else:
            porsis.append(0)
    return round(min(porsis) if porsis else 0, 2)

# Rekomendasi utama (harus ada bahannya)
def recommend_recipe(user_input):
    if knn is None or len(ingredient_columns) == 0:
        return []
    
    results = []
    user_vector = np.array([user_input.get(i, 0) for i in ingredient_columns]).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector)

    for idx in indices[0]:
        resep = df.iloc[idx]
        recipe_parsed = resep['Parsed']
        
        common_ingredients = [b for b in recipe_parsed if b in user_input]
        if not common_ingredients:
            continue
        
        max_porsi = calculate_portions(user_input, recipe_parsed)
        if max_porsi <= 0:
            continue

        results.append({
            'title': resep['Title'],
            'ingredients': recipe_parsed,
            'steps': resep.get('Steps', 'No steps available'),
            'porsi': max_porsi,
            'distance': distances[0][len(results)]
        })
    
    return sorted(results, key=lambda x: x['porsi'], reverse=True)

# Rekomendasi dengan bahan tambahan
def recommend_with_additional_ingredients(user_input):
    if knn is None or len(ingredient_columns) == 0:
        return []
    
    results = []
    user_vector = np.array([user_input.get(i, 0) for i in ingredient_columns]).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector)

    for idx in indices[0]:
        resep = df.iloc[idx]
        recipe_parsed = resep['Parsed']
        
        missing = [b for b in recipe_parsed if b not in user_input]
        if missing:
            results.append({
                'title': resep['Title'],
                'missing': missing,
                'steps': resep.get('Steps', 'No steps available')
            })
    
    return results

# === ROUTE FLASK ===
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    additional_recommendations = []
    if request.method == 'POST':
        names = request.form.getlist('ingredient_name[]')
        qtys = request.form.getlist('ingredient_qty[]')
        
        user_stock = {}
        for name, qty in zip(names, qtys):
            if name.strip() and qty.strip():
                try:
                    user_stock[name.lower().strip()] = float(qty)
                except ValueError:
                    continue
        
        if user_stock:
            recommendations = recommend_recipe(user_stock)
            show_additional = request.form.get('show_additional_recipes') == 'on'
            if show_additional:
                additional_recommendations = recommend_with_additional_ingredients(user_stock)
    
    return render_template('index.html',
                           recommendations=recommendations,
                           additional_recommendations=additional_recommendations)

# === RUN SERVER ===
if __name__ == '__main__':
    app.run(debug=True)
