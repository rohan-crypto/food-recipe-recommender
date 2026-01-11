import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import recommend_recipes, ingredient_overlap, is_veg, is_vegan
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
try:
    from nltk.corpus import stopwords
    EN_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    EN_STOPWORDS = set(stopwords.words('english'))

# NLTK stopwords
EN_STOPWORDS = set(stopwords.words('english'))

RECIPE_STOPWORDS = {
    # Units
    "cup", "cups", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
    "tbsp", "tsp", "ounce", "ounces", "oz", "pound", "pounds", "g", "kg",
    "ml", "l", "liter", "liters", "pinch", "clove", "cloves", "slice", "slices",
    # Numbers / fractions
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "½", "¼", "¾", "one", "two", "three", "four", "five", "six",
    # Sizes / descriptors
    "small", "medium", "large", "fresh", "ground", "minced", "diced",
    # Miscellaneous
    "back", "away", "baby", "ablespoon", "ablespoons", "bags", "bag", "spray",
    "freshly", "optional", "taste", "seasoning", "drained", "divided", "extra",
    "unsalted", "salted", "hot", "cold", "warm", "room", "temperature",
    # Prepositions / connectors
    "and", "or", "of", "to", "with", "in", "for", "on", "from",
}

STOPWORDS = EN_STOPWORDS | RECIPE_STOPWORDS

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Recipes.csv")

    # Create clean_ingredients for matching
    df["clean_ingredients"] = df["ingredients"].astype(str).apply(
        lambda x: [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', x) if w.lower() not in STOPWORDS]
    )

    # Keep ingredient_list for displaying in recipes
    df["ingredient_list"] = df["ingredients"].astype(str).apply(lambda x: [i.strip() for i in x.split(",")])
    
    return df

df = load_data()

# Initialize TF-IDF
@st.cache_resource
def init_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["clean_ingredients"].apply(lambda x: " ".join(x)))
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = init_tfidf(df)

# Streamlit page
st.set_page_config(page_title="Recipe Recommender", layout="wide")
st.title("Recipe Recommender with Semantic Match")
st.markdown("Enter the ingredients you have, and find the best matching recipes!")

# Sidebar
st.sidebar.header("Filters and Weights")

# Sidebar ingredients list
all_words = [i for sublist in df["clean_ingredients"] for i in sublist]
word_counts = Counter(all_words)
filtered_ingredients = sorted(set(w for w in word_counts if len(w) > 2 and word_counts[w] >= 2))
all_ingredients = filtered_ingredients

st.sidebar.markdown("### Select ingredients (autocomplete simulation)")
user_ingredients_list = st.sidebar.multiselect(
    "Choose ingredients you have:",
    options=all_ingredients,
    default=[]
)

user_ingredients = " ".join(user_ingredients_list)

# Recommendation controls
top_n = st.sidebar.slider("Number of recipes to show", 1, 20, 5)
fetch_k = st.sidebar.slider("Initial fetch size", 5, 50, 20)
alpha = st.sidebar.slider("Weight: semantic vs ingredients", 0.0, 1.0, 0.7, 0.05)
min_similarity = st.sidebar.slider("Min similarity threshold", 0.0, 0.5, 0.05, 0.01)

# Filters
max_steps = st.sidebar.slider("Max steps (optional)", 1, 20, 5)
max_ingredients = st.sidebar.slider("Max ingredients (optional)", 1, 20, 8)
veg = st.sidebar.checkbox("Vegetarian only")
vegan = st.sidebar.checkbox("Vegan only")

# Sorting
sort_by = st.sidebar.selectbox(
    "Sort recommended recipes by:",
    options=["final_score", "num_matched_ingredients"],
    index=0,
    format_func=lambda x: "Final weighted score" if x=="final_score" else "Number of matched ingredients"
)

# Horizontal bar
def render_bar(score, label, width=200):
    pct = score * 100
    color = "#4CAF50" if pct >= 70 else "#FFC107" if pct >= 40 else "#F44336"
    bar_html = f"""
    <div style='border:1px solid #ccc; width:{width}px; height:15px;'>
        <div style='background-color:{color}; width:{pct}%; height:15px;'></div>
    </div>
    <small>{label}: {score:.3f}</small>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

# Find recipes
if st.button("Find Recipes"):
    if not user_ingredients_list:
        st.warning("Please select at least one ingredient!")
    else:
        with st.spinner("Finding best recipes..."):
            recommended = recommend_recipes(
                user_ingredients=user_ingredients,
                tfidf=tfidf,
                tfidf_matrix=tfidf_matrix,
                df=df,
                top_n=top_n,
                fetch_k=fetch_k,
                max_steps=max_steps,
                max_ingredients=max_ingredients,
                veg=veg,
                vegan=vegan,
                alpha=alpha,
                min_similarity=min_similarity
            )

            ascending = False if sort_by=="final_score" else True
            recommended = recommended.sort_values(by=sort_by, ascending=ascending)

        if recommended.empty:
            st.info("Sorry, no recipes matched your ingredients and filters")
        else:
            st.success(f"Found {len(recommended)} recipes!")

            for _, row in recommended.iterrows():
                with st.expander(f"{row['recipe_title']} ({row['category']})"):
                    st.write(row['description'])

                    st.markdown("**Ingredients:**")
                    ing_text = ", ".join(row["ingredient_list"])  # display original phrases
                    for ing in row["matched_ingredients"]:
                        ing_text = re.sub(rf"\b{re.escape(ing)}\b", f"**{ing}**", ing_text, flags=re.IGNORECASE)
                    st.markdown(ing_text)

                    st.markdown(
                        f"**Steps:** {row['num_steps']} | "
                        f"**Ingredients:** {row['num_ingredients']} | "
                        f"**Matched:** {row['num_matched_ingredients']} / {len(user_ingredients_list)}"
                    )

                    render_bar(row["match_score"], "Semantic match score")
                    render_bar(row["ingredient_overlap_score"], "Ingredient overlap score")
                    render_bar(row["final_score"], "Final score")

                    if veg: st.markdown("Vegetarian")
                    if vegan: st.markdown("Vegan")

            csv = recommended.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Recipes as CSV",
                data=csv,
                file_name="recommended_recipes.csv",
                mime="text/csv"
            )
