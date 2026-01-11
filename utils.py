import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

NON_VEG_ITEMS = {
    "chicken", "mutton", "beef", "pork", "fish", "tuna", "egg", "eggs",
    "shrimp", "prawn", "crab", "lobster", "bacon", "ham", "turkey",
    "rattlesnake", "duck", "goat", "anchovy", "salmon", "sardine"
}
DAIRY_ITEMS = {
    "milk", "cheese", "butter", "cream", "yogurt", "curd", "ghee",
    "egg", "eggs", "honey", "parmesan", "mozzarella", "cheddar"
}

def ingredient_overlap(user_ingredients, recipe_ingredients):
    user_set = set(user_ingredients.lower().split())
    recipe_set = set(recipe_ingredients)  # already cleaned
    return list(user_set & recipe_set)

def is_veg(ingredients):
    return not any(i.lower() in NON_VEG_ITEMS for i in ingredients)

def is_vegan(ingredients):
    forbidden = NON_VEG_ITEMS | DAIRY_ITEMS
    return not any(i.lower() in forbidden for i in ingredients)

def recommend_recipes(
    user_ingredients, tfidf, tfidf_matrix, df,
    top_n=5, fetch_k=20, max_steps=None, max_ingredients=None,
    veg=False, vegan=False, alpha=0.7, min_similarity=0.05
):
    user_vec = tfidf.transform([user_ingredients.lower()])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-fetch_k:][::-1]
    recommended = df.iloc[top_indices].copy()
    recommended["match_score"] = similarities[top_indices].round(3)

    # Remove duplicates
    recommended = recommended.drop_duplicates(subset="recipe_title")

    # Matched ingredients
    recommended["matched_ingredients"] = recommended["clean_ingredients"].apply(
        lambda x: ingredient_overlap(user_ingredients, x)
    )
    recommended["num_matched_ingredients"] = recommended["matched_ingredients"].apply(len)
    user_count = max(1, len(user_ingredients.split()))
    recommended["ingredient_overlap_score"] = recommended["num_matched_ingredients"] / user_count

    # Final score
    recommended["final_score"] = alpha*recommended["match_score"] + (1-alpha)*recommended["ingredient_overlap_score"]

    # Filters
    if max_steps: recommended = recommended[recommended["num_steps"] <= max_steps]
    if max_ingredients: recommended = recommended[recommended["num_ingredients"] <= max_ingredients]
    if veg: recommended = recommended[recommended["clean_ingredients"].apply(is_veg)]
    if vegan: recommended = recommended[recommended["clean_ingredients"].apply(is_vegan)]
    recommended = recommended[recommended["match_score"] >= min_similarity]

    return recommended.head(top_n)
