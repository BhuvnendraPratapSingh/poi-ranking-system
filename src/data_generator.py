"""
generates synthetic datasets for the POI ranking system.
three CSVs: pois, travelers, interactions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, os

DESTINATION = "Paris"
CENTER_LAT, CENTER_LON = 48.8566, 2.3522

CATEGORIES = {
    "Restaurant": ["Fine Dining", "Bistro", "Cafe", "Street Food", "Bakery",
                    "Wine Bar", "Local Cuisine", "International"],
    "Museum": ["Art Museum", "History Museum", "Science Museum",
               "Contemporary Art", "Specialty Museum"],
    "Historical Site": ["Monument", "Church/Cathedral", "Palace",
                        "Archaeological Site", "Historic District"],
    "Nature/Outdoor": ["Park", "Garden", "River Walk", "Viewpoint", "Forest"],
    "Shopping": ["Market", "Boutique", "Flea Market", "Department Store", "Artisan Shop"],
    "Activity": ["Walking Tour", "Cooking Class", "Boat Tour", "Bike Tour",
                 "Workshop", "Show/Performance"],
    "Neighborhood": ["Historic Quarter", "Art District", "Local Village",
                     "Bohemian Area", "Waterfront"],
    "Nightlife": ["Jazz Club", "Cocktail Bar", "Live Music", "Theater", "Cabaret"],
    "Family": ["Theme Park", "Aquarium", "Zoo", "Interactive Museum", "Playground"],
    "Event": ["Festival", "Exhibition", "Concert", "Food Event", "Cultural Event"],
}

# neighborhoods with lat/lon - used to cluster POIs geographically
HOODS = [
    ("Montmartre", 48.8867, 2.3431), ("Le Marais", 48.8566, 2.3622),
    ("Latin Quarter", 48.8495, 2.3471), ("Saint-Germain", 48.8539, 2.3349),
    ("Champs-Élysées", 48.8698, 2.3075), ("Bastille", 48.8533, 2.3692),
    ("Belleville", 48.8718, 2.3842), ("Oberkampf", 48.8650, 2.3780),
    ("Canal Saint-Martin", 48.8710, 2.3630), ("Trocadéro", 48.8627, 2.2876),
    ("Opéra", 48.8710, 2.3320), ("Les Halles", 48.8613, 2.3470),
    ("Pigalle", 48.8821, 2.3375), ("Île de la Cité", 48.8552, 2.3477),
    ("Bercy", 48.8340, 2.3876),
]

TAG_POOL = {
    "atmosphere": ["romantic", "lively", "quiet", "historic", "modern", "cozy",
                   "elegant", "rustic", "bohemian", "family-friendly", "trendy",
                   "traditional", "intimate", "vibrant"],
    "crowd": ["touristy", "local-favorite", "hidden-gem", "popular",
              "off-beaten-path", "well-known", "undiscovered"],
    "accessibility": ["wheelchair-accessible", "stroller-friendly", "easy-access",
                      "requires-walking", "hilly-terrain"],
    "special": ["instagram-worthy", "michelin-starred", "unesco-heritage",
                "free-entry", "reservation-required", "seasonal",
                "rain-friendly", "outdoor-only", "kid-friendly",
                "pet-friendly", "group-friendly"],
}

INTERESTS = [
    "local food", "fine dining", "street food", "history", "architecture",
    "museums", "art", "contemporary art", "nature", "parks", "neighborhoods",
    "local experiences", "shopping", "markets", "nightlife", "music",
    "theater", "photography", "walking", "cycling", "family activities",
    "wine", "cooking", "festivals", "sports", "wellness",
]

# weights for different interaction types
# positive = user liked it, negative = nah
INTERACTION_WEIGHTS = {
    "view": 1.0, "click": 2.0, "save": 4.0, "share": 4.5,
    "navigate": 5.0, "visit": 7.0, "booking": 8.0, "dismiss": -2.0,
}


def _jitter_geo(center, spread=8.0, rng=None):
    """add gaussian noise to a lat/lon point"""
    rng = rng or np.random.default_rng()
    dlat = rng.normal(0, spread * 0.009 / 2)
    dlon = rng.normal(0, spread * 0.009 / 2)
    return center[0] + dlat, center[1] + dlon


def _make_name(cat, subcat, idx, rng):
    prefixes = {
        "Restaurant": ["Le", "La", "Chez", "Au", "L'"],
        "Museum": ["Musée", "Galerie", "Centre"],
        "Historical Site": ["", "Place", "Tour", "Pont"],
        "Nature/Outdoor": ["Jardin", "Parc", "Promenade"],
        "Shopping": ["Marché", "Boutique", "Atelier"],
        "Activity": ["", "Paris", ""],
        "Neighborhood": ["Quartier", "Village", "Passage"],
        "Nightlife": ["Le", "La", "Club"],
        "Family": ["", "Parc", ""],
        "Event": ["Festival", "Salon", ""],
    }
    words = ["Lumière", "Étoile", "Fleur", "Seine", "Cœur", "Pierre",
             "Bleu", "Rose", "Soleil", "Vent", "Vigne", "Plume",
             "Petit", "Grand", "Ancien", "Nouveau", "Beau", "Doré",
             "Marcel", "Hugo", "Voltaire", "Molière", "Renoir", "Monet"]
    
    pre = rng.choice(prefixes.get(cat, [""]))
    w = rng.choice(words)
    suf = subcat.split()[0] if rng.random() < 0.3 else ""
    name = f"{pre} {w} {suf}".strip()
    return name or f"Paris {subcat} #{idx}"


def _make_desc(cat, subcat, tags, rng):
    tmpls = {
        "Restaurant": [
            "A charming {s} offering authentic flavors in a {a} setting.",
            "Popular {s} known for fresh ingredients and {a} ambiance.",
            "Delightful {s} serving classic and creative dishes.",
        ],
        "Museum": [
            "Fascinating {s} showcasing remarkable collections.",
            "An inspiring {s} with permanent and rotating exhibitions.",
            "Renowned {s} featuring works spanning centuries.",
        ],
        "Historical Site": [
            "A significant {s} rich in history and architectural beauty.",
            "Iconic {s} dating back centuries.",
            "Impressive {s} that captures Paris's storied heritage.",
        ],
        "Nature/Outdoor": [
            "Beautiful {s} perfect for relaxation and scenic views.",
            "Serene {s} offering an escape from the city bustle.",
            "Lovely {s} popular with locals and visitors.",
        ],
        "Shopping": [
            "Vibrant {s} with unique finds and local goods.",
            "Charming {s} offering artisanal products.",
            "Popular {s} with an eclectic mix of vendors.",
        ],
        "Activity": [
            "Engaging {s} providing a hands-on experience.",
            "Exciting {s} for the adventurous and curious.",
            "Fun {s} offering a unique way to explore Paris.",
        ],
        "Neighborhood": [
            "A {a} neighborhood with distinctive charm.",
            "Explore the {a} streets and discover local treasures.",
            "Known for its {a} vibe and authentic culture.",
        ],
        "Nightlife": [
            "A {a} venue known for great music.",
            "Popular {s} with locals for memorable evenings.",
            "Exciting {s} that comes alive after dark.",
        ],
        "Family": [
            "Family-friendly {s} with activities for all ages.",
            "Fun {s} that children and parents both enjoy.",
            "Interactive {s} designed to educate and entertain.",
        ],
        "Event": [
            "Seasonal {s} showcasing Parisian culture.",
            "Popular {s} drawing enthusiasts worldwide.",
            "Annual {s} celebrating local traditions.",
        ],
    }
    a = tags[0] if tags else "pleasant"
    t = rng.choice(tmpls.get(cat, ["A notable place to visit."]))
    return t.format(s=subcat.lower(), a=a)


def generate_pois(n=500, seed=42):
    rng = np.random.default_rng(seed)
    
    cat_weights = {
        "Restaurant": 0.22, "Museum": 0.08, "Historical Site": 0.10,
        "Nature/Outdoor": 0.08, "Shopping": 0.10, "Activity": 0.12,
        "Neighborhood": 0.06, "Nightlife": 0.07, "Family": 0.07, "Event": 0.10,
    }
    cats = list(cat_weights.keys())
    probs = [cat_weights[c] for c in cats]
    
    rows = []
    for i in range(n):
        cat = rng.choice(cats, p=probs)
        subcat = rng.choice(CATEGORIES[cat])
        
        hood = HOODS[rng.integers(len(HOODS))]
        lat, lon = _jitter_geo((hood[1], hood[2]), spread=1.5, rng=rng)
        
        # popularity is power-law distributed (most POIs aren't that popular)
        pop_raw = rng.pareto(a=1.5) + 1
        pop = min(pop_raw / 10.0, 1.0)
        
        rating = np.clip(3.0 + pop * 1.5 + rng.normal(0, 0.4), 1.0, 5.0)
        n_reviews = int(max(1, rng.poisson(pop * 500 + 10)))
        
        # price depends on category
        if cat in ("Restaurant", "Nightlife"):
            price = rng.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.30, 0.20])
        elif cat in ("Shopping", "Activity"):
            price = rng.choice([1, 2, 3, 4], p=[0.20, 0.40, 0.25, 0.15])
        else:
            price = rng.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.20, 0.10])
        
        dur_ranges = {
            "Restaurant": (45, 120), "Museum": (60, 180), "Historical Site": (30, 90),
            "Nature/Outdoor": (30, 120), "Shopping": (30, 90), "Activity": (60, 180),
            "Neighborhood": (60, 180), "Nightlife": (60, 180), "Family": (60, 180),
            "Event": (60, 240),
        }
        lo, hi = dur_ranges[cat]
        duration = int(rng.integers(lo, hi + 1))
        
        # opening hours
        if cat == "Nightlife":
            oh, ch = rng.choice([18, 19, 20]), rng.choice([1, 2, 3])
        else:
            oh, ch = rng.choice([7, 8, 9, 10, 11]), rng.choice([17, 18, 19, 20, 21, 22, 23])
        hours = f"{oh:02d}:00-{ch:02d}:00"
        
        needs_reservation = bool(rng.random() < (0.4 if cat in ("Restaurant", "Activity") else 0.1))
        access = rng.choice(["fully-accessible", "partially-accessible", "limited-access"],
                            p=[0.5, 0.35, 0.15])
        
        # build tags
        tags = []
        for grp in TAG_POOL.values():
            if rng.random() < 0.7:
                tags.append(rng.choice(grp))
        tags = list(set(tags))[:rng.integers(3, 8)]
        
        # popular stuff tends to be touristy, unpopular stuff tends to be local
        if pop > 0.6:
            tags = [t for t in tags if t not in ("local-favorite", "hidden-gem")]
            if "touristy" not in tags:
                tags.append("touristy")
        elif pop < 0.2:
            tags = [t for t in tags if t != "touristy"]
            if rng.random() < 0.5:
                tags.append(rng.choice(["local-favorite", "hidden-gem", "off-beaten-path"]))
        
        rows.append({
            "poi_id": f"P{i+1:04d}",
            "name": _make_name(cat, subcat, i, rng),
            "category": cat, "subcategory": subcat,
            "latitude": round(lat, 6), "longitude": round(lon, 6),
            "neighborhood": hood[0],
            "description": _make_desc(cat, subcat, tags, rng),
            "tags": json.dumps(tags),
            "price_level": int(price),
            "rating": round(rating, 2),
            "review_count": n_reviews,
            "popularity": round(pop, 4),
            "expected_duration": duration,
            "opening_hours": hours,
            "reservation_required": needs_reservation,
            "accessibility": access,
        })
    
    return pd.DataFrame(rows)


def generate_travelers(n=50, seed=42):
    rng = np.random.default_rng(seed)
    
    # a few archetypes to make the data interesting
    archetypes = [
        {"interests": ["local food", "neighborhoods", "local experiences", "markets", "street food"],
         "budget": "medium", "pref": "Prefer less touristy, authentic local experiences"},
        {"interests": ["history", "architecture", "museums", "art"],
         "budget": "high", "pref": "Interested in famous landmarks and cultural heritage"},
        {"interests": ["family activities", "parks", "nature"],
         "budget": "medium", "pref": "Looking for child-friendly, interactive experiences"},
        {"interests": ["nightlife", "music", "fine dining", "wine"],
         "budget": "high", "pref": "Enjoy upscale dining and vibrant nightlife"},
        {"interests": ["nature", "walking", "cycling", "photography"],
         "budget": "low", "pref": "Enjoy outdoor activities and scenic walks"},
        {"interests": ["shopping", "art", "contemporary art", "cooking"],
         "budget": "medium", "pref": "Enjoy creative and artisanal experiences"},
    ]
    
    rows = []
    for i in range(n):
        arch = archetypes[i] if i < len(archetypes) else archetypes[rng.integers(len(archetypes))]
        
        interests = list(arch["interests"])
        # sprinkle in some random extra interests
        for _ in range(rng.integers(0, 3)):
            extra = rng.choice(INTERESTS)
            if extra not in interests:
                interests.append(extra)
        
        start = datetime(2025, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        dur = int(rng.choice([2, 3, 4, 5, 7, 10, 14]))
        end = start + timedelta(days=dur)
        
        budget = arch["budget"] if rng.random() < 0.7 else rng.choice(["low", "medium", "high"])
        
        party = rng.choice(["solo", "couple", "family", "friends", "business"])
        if "family activities" in interests:
            party = "family"
        
        mobility = rng.choice(["walking", "public_transport", "car", "bicycle"],
                              p=[0.30, 0.40, 0.15, 0.15])
        if party == "family":
            mobility = rng.choice(["walking", "public_transport", "car"], p=[0.2, 0.4, 0.4])
        
        pref_text = arch["pref"]
        if rng.random() < 0.3:
            pref_text += ". " + rng.choice([
                "Avoid very crowded places",
                "Prefer morning activities",
                "Would like at least one unique experience",
                "Open to suggestions outside main interests",
                "Prefer walkable distances between places",
            ])
        
        rows.append({
            "traveler_id": f"U{i+1:04d}",
            "destination": DESTINATION,
            "travel_start": start.strftime("%Y-%m-%d"),
            "travel_end": end.strftime("%Y-%m-%d"),
            "trip_duration": dur,
            "interests": json.dumps(interests),
            "budget": budget,
            "party_type": party,
            "mobility": mobility,
            "explicit_preferences": pref_text,
        })
    
    return pd.DataFrame(rows)


def generate_interactions(poi_df, trav_df, n=5000, seed=42):
    """
    interactions are biased toward preference-matching --
    travelers interact more with POIs that fit their interests.
    this is what lets the model learn actual patterns.
    """
    rng = np.random.default_rng(seed)
    
    itypes = list(INTERACTION_WEIGHTS.keys())
    
    # which interests map to which categories
    cat_interest = {
        "Restaurant": ["local food", "fine dining", "street food", "wine", "cooking"],
        "Museum": ["museums", "art", "contemporary art", "history"],
        "Historical Site": ["history", "architecture", "photography"],
        "Nature/Outdoor": ["nature", "parks", "walking", "cycling", "photography"],
        "Shopping": ["shopping", "markets"],
        "Activity": ["local experiences", "cooking", "cycling", "walking"],
        "Neighborhood": ["neighborhoods", "local experiences", "photography"],
        "Nightlife": ["nightlife", "music", "wine"],
        "Family": ["family activities", "parks"],
        "Event": ["festivals", "music", "local experiences"],
    }
    
    rows = []
    for _ in range(n):
        trow = trav_df.iloc[rng.integers(len(trav_df))]
        tid = trow["traveler_id"]
        tinterests = set(json.loads(trow["interests"]))
        
        # compute how well each POI matches this traveler
        scores = []
        for _, p in poi_df.iterrows():
            ci = set(cat_interest.get(p["category"], []))
            match = len(tinterests & ci)
            scores.append(match * 3.0 + p["popularity"] * 2.0 + 0.5)
        
        probs = np.array(scores)
        probs /= probs.sum()
        
        pidx = rng.choice(len(poi_df), p=probs)
        prow = poi_df.iloc[pidx]
        
        # interaction type depends on how good the match is
        ci = set(cat_interest.get(prow["category"], []))
        match = len(tinterests & ci)
        
        if match >= 2:
            # good match -> deeper engagement
            itype = rng.choice(itypes, p=[0.10, 0.10, 0.15, 0.10, 0.15, 0.25, 0.10, 0.05])
        elif match == 1:
            itype = rng.choice(itypes, p=[0.25, 0.20, 0.15, 0.08, 0.10, 0.12, 0.05, 0.05])
        else:
            # bad match -> mostly just views + dismissals
            itype = rng.choice(itypes, p=[0.35, 0.20, 0.05, 0.02, 0.05, 0.05, 0.02, 0.26])
        
        base = datetime.strptime(trow["travel_start"], "%Y-%m-%d")
        ts = base + timedelta(days=int(rng.integers(-30, trow["trip_duration"] + 1)))
        
        rows.append({
            "traveler_id": tid,
            "poi_id": prow["poi_id"],
            "interaction_type": itype,
            "interaction_weight": INTERACTION_WEIGHTS[itype],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    return pd.DataFrame(rows)


def generate_all(out_dir="data", n_pois=500, n_travelers=50, n_interactions=5000, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    
    print("generating POIs...")
    pois = generate_pois(n_pois, seed)
    pois.to_csv(os.path.join(out_dir, "pois.csv"), index=False)
    print(f"  {len(pois)} pois")
    
    print("generating travelers...")
    travelers = generate_travelers(n_travelers, seed)
    travelers.to_csv(os.path.join(out_dir, "travelers.csv"), index=False)
    print(f"  {len(travelers)} travelers")
    
    print("generating interactions...")
    interactions = generate_interactions(pois, travelers, n_interactions, seed)
    interactions.to_csv(os.path.join(out_dir, "interactions.csv"), index=False)
    print(f"  {len(interactions)} interactions")
    
    return pois, travelers, interactions


if __name__ == "__main__":
    generate_all()
