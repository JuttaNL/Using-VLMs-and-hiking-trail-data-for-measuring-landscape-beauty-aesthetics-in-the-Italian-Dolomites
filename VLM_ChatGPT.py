import geopandas as gpd
import pandas as pd
import torch
import numpy as np
import sqlite3
import unicodedata  # 🔧 Accent removal
from transformers import AutoProcessor, AutoModel
from pathlib import Path

# === 🔧 Utility: Remove accents from strings ===
def remove_accents(input_str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', input_str)
        if unicodedata.category(c) != 'Mn'
    )

# === Load and normalize GeoDataFrame ===
son_pts = gpd.read_file("/Users/just/Documents/ERM/Thesis stuff/GIS Files/Trail_image_correct_image_ID.geojson")
son_pts["id"] = (
    son_pts["id"]
    .apply(remove_accents)  # 🔧 Remove accents
    .str.replace(",", "", regex=False)
    .str.replace(":", "", regex=False)  # 🔧 Remove colons
    .str.replace(" - ", " ", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace(r"_(\d+)$", r"_image_\1", regex=True)
    .str.lower()
)
son_pts.set_index("id", inplace=True)

# === Load text prompts and values ===
all_prompts = pd.read_csv("/Users/just/Documents/ERM/Thesis stuff/dolomites_50_landscape_descriptions_ChatGPT.csv")
prompts_in_order = list(all_prompts['prompts_in_order'].values)
values_in_order = torch.tensor(all_prompts['values_in_order'].values, dtype=torch.float32)

# === Load text encoder and processor ===
encoder = "siglip2"
if encoder == "siglip2":
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384",
        do_rescale=True,
        use_fast=True
    )
model.requires_grad_(False)

# === Load embeddings from SQLite ===
def load_embeddings_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT folder_id, image_id, embedding FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    embeddings = {}
    for folder_id, image_id, embedding_blob in rows:
        folder = (
            remove_accents(folder_id)
            .replace(",", "")
            .replace(":", "")  # 🔧 Remove colons
            .replace(" - ", " ")
            .strip().lower().replace(" ", "_")
        )
        image = str(image_id).strip().lower().replace(" ", "_")
        combined_id = f"{folder}_{image}"
        embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
        embedding_tensor = torch.from_numpy(embedding_array)
        embeddings[combined_id] = embedding_tensor
    return embeddings

db_path = "/Users/just/Documents/ERM/Thesis stuff/hiking_embeddings/embeds.db"
embeddings = load_embeddings_from_db(db_path)

# === Match embeddings to GeoDataFrame entries ===
embeds = []
embeds_keys = []
for k, v in embeddings.items():
    if k in son_pts.index:
        embeds.append(v)
        embeds_keys.append(k)

print(f"✅ Number of embeddings matched: {len(embeds)}")

# === Encode text prompts ===
inputs = processor(
    text=prompts_in_order,
    images=None,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
with torch.no_grad():
    text_embeds = model.get_text_features(**inputs)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# === Inference ===
emb_index = 0
early_ensembling_preds = []

with torch.no_grad():
    while emb_index < len(embeds):
        upper_index = min(emb_index + 20000, len(embeds))
        batch_embeddings = embeds[emb_index:upper_index]
        image_embeds = torch.stack(batch_embeddings)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias
        activations = torch.softmax(logits_per_text, dim=0)
        early_preds = torch.sum(activations * values_in_order.unsqueeze(1), dim=0)
        early_ensembling_preds.append(early_preds)

        print(f"🔄 Processed embeddings {emb_index} to {upper_index}")
        emb_index += 20000

# === Assign predictions to matched subset ===
if early_ensembling_preds:
    early_preds_all = torch.cat(early_ensembling_preds)
    son_pts_matched = son_pts.loc[embeds_keys]
    son_pts_matched["early_ensemble_pred"] = early_preds_all.numpy()
else:
    son_pts_matched = son_pts.loc[[]]

# === Export predictions ===
output_dir = Path("data/outputs/ensembling/")
output_dir.mkdir(parents=True, exist_ok=True)
son_pts_matched.to_file(output_dir / "ensembling_preds_no_gt.gpkg", driver="GPKG")

output_path = Path("/Users/just/Documents/ERM/Thesis stuff/Output of LPE ChatGPT/ensembling_preds_no_gt.gpkg")
output_path.parent.mkdir(parents=True, exist_ok=True)
son_pts_matched.to_file(output_path, driver="GPKG")

# === Final report ===
print(f"\n✅ Successfully exported matched predictions to: {output_path}")
print(f"📍 File contains {len(son_pts_matched)} matched points")
print(f"📦 Number of prediction batches: {len(early_ensembling_preds)}")
if early_ensembling_preds:
    print(f"📈 Shape of final predictions: {early_preds_all.shape}")