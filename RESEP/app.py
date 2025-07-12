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

# Fungsi untuk membersihkan dan parsing ingredients
def clean_ingredient_text(s):
    try:
        if isinstance(s, str):
            # Coba parsing sebagai literal Python dulu
            try:
                return ast.literal_eval(s)
            except:
                # Jika gagal, coba parsing manual
                # Hilangkan bracket dan quote
                s = s.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1]
                
                # Split by comma dan bersihkan
                ingredients = []
                for item in s.split(','):
                    item = item.strip().strip('"\'')
                    if item:
                        ingredients.append(item)
                return ingredients
        return []
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Original string: {s}")
        return []

df['Ingredients'] = df['Ingredients'].apply(clean_ingredient_text)

# Parsing kuantitas bahan
def parse_ingredients(ingredient_list):
    ingredient_dict = {}
    
    if not ingredient_list:
        return ingredient_dict
    
    for item in ingredient_list:
        if not isinstance(item, str):
            continue
            
        item = item.strip()
        if not item:
            continue
            
        # Pattern untuk menangkap angka, unit, dan nama bahan
        # Contoh: "2 sendok teh garam", "1/2 kg beras", "3 butir telur"
        patterns = [
            r"(\d+(?:[.,/]\d+)?)\s*(kg|gram|gr|g|sendok|sdt|sdm|butir|biji|lembar|potong|siung|buah|cup|ml|liter|l)\s+(.*)",
            r"(\d+(?:[.,/]\d+)?)\s+(.*)",  # Tanpa unit
            r"(.*)"  # Fallback untuk nama saja
        ]
        
        parsed = False
        for pattern in patterns:
            match = re.match(pattern, item.lower().strip())
            if match:
                groups = match.groups()
                
                if len(groups) >= 2:  # Ada quantity
                    try:
                        qty_str = groups[0].replace(',', '.')
                        
                        # Handle fraction
                        if '/' in qty_str:
                            parts = qty_str.split('/')
                            if len(parts) == 2:
                                qty = float(parts[0]) / float(parts[1])
                            else:
                                qty = float(qty_str.replace('/', '.'))
                        else:
                            qty = float(qty_str)
                        
                        unit = groups[1] if len(groups) > 2 else ''
                        name = groups[2] if len(groups) > 2 else groups[1]
                        
                        # Konversi unit ke gram
                        if 'kg' in unit:
                            qty *= 1000
                        elif 'butir' in unit or 'biji' in unit:
                            qty *= 50  # Asumsi 1 butir = 50g
                        elif 'sendok' in unit or 'sdt' in unit:
                            qty *= 5   # 1 sendok teh = 5g
                        elif 'sdm' in unit:
                            qty *= 15  # 1 sendok makan = 15g
                        elif 'siung' in unit:
                            qty *= 3   # 1 siung bawang = 3g
                        elif 'lembar' in unit:
                            qty *= 10  # 1 lembar = 10g
                        elif 'potong' in unit:
                            qty *= 100 # 1 potong = 100g
                        elif 'buah' in unit:
                            qty *= 100 # 1 buah = 100g
                        elif 'cup' in unit:
                            qty *= 240 # 1 cup = 240g
                        elif 'ml' in unit:
                            qty *= 1   # 1 ml = 1g (for liquid)
                        elif 'liter' in unit or 'l' in unit:
                            qty *= 1000
                        
                        name = name.strip()
                        if name:
                            ingredient_dict[name] = qty
                            parsed = True
                            break
                    except ValueError:
                        continue
                else:  # Hanya nama bahan
                    name = groups[0].strip()
                    if name:
                        ingredient_dict[name] = 100  # Default 100g
                        parsed = True
                        break
        
        if not parsed:
            # Fallback: masukkan sebagai nama dengan quantity default
            clean_name = item.strip()
            if clean_name:
                ingredient_dict[clean_name] = 100
    
    return ingredient_dict

df['Parsed'] = df['Ingredients'].apply(parse_ingredients)

# Debug output
print("Contoh baris Ingredients:", df['Ingredients'].iloc[0] if len(df) > 0 else "No data")
print("Parsed hasilnya:", df['Parsed'].iloc[0] if len(df) > 0 else "No data")

# Ambil semua nama bahan sebagai kolom vektor
all_ingredients = set()
for parsed in df['Parsed']:
    if parsed:  # Pastikan parsed tidak kosong
        all_ingredients.update(parsed.keys())

ingredient_columns = list(all_ingredients)
print(f"Total unique ingredients: {len(ingredient_columns)}")

if len(ingredient_columns) > 0:
    print(f"Sample ingredients: {ingredient_columns[:5]}")

def ingredient_to_vector(parsed):
    return [parsed.get(i, 0) for i in ingredient_columns]

X_ingredients = np.array([ingredient_to_vector(p) for p in df['Parsed']])

# Latih model KNN
if len(ingredient_columns) > 0:
    knn = NearestNeighbors(n_neighbors=min(3, len(df)), metric='euclidean')
    knn.fit(X_ingredients)
    print("KNN model trained successfully")
else:
    print("ERROR: No ingredients found to train model")
    knn = None

# === FUNGSI REKOMENDASI ===
def calculate_portions(user_stock, recipe_stock):
    porsis = []
    for bahan, qty in recipe_stock.items():
        if bahan in user_stock and qty > 0:
            porsis.append(user_stock[bahan] / qty)
        else:
            porsis.append(0)
    return round(min(porsis) if porsis else 0, 2)

def recommend_recipe(user_input):
    if knn is None or len(ingredient_columns) == 0:
        return []
    
    user_vector = np.array([user_input.get(i, 0) for i in ingredient_columns]).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector)
    
    results = []
    for idx in indices[0]:
        resep = df.iloc[idx]
        max_porsi = calculate_portions(user_input, resep['Parsed'])
        results.append({
            'title': resep['Title'],
            'ingredients': resep['Parsed'],
            'steps': resep.get('Steps', 'No steps available'),
            'porsi': max_porsi,
            'distance': distances[0][len(results)]
        })
    return results

# === ROUTE FLASK ===
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
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
            print("Input user:", user_stock)
            print("Rekomendasi ditemukan:", len(recommendations))
        else:
            print("No valid user input")
    
    return render_template('index.html', recommendations=recommendations)

# === RUN SERVER ===
if __name__ == '__main__':
    app.run(debug=True)