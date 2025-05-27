import sqlite3
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from transformers import AutoProcessor, AutoModel
from codebase.utils.file_utils import recursively_get_files
from codebase.utils.downloader_utils import ImageHandler

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##### SET GLOBAL OPTIONS ######
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

# https://huggingface.co/google/siglip-so400m-patch14-384

# Initialize SQLite database
db_file = "embeds.db"
image_folder = f"/Users/just/Documents/ERM/Hiking trail images"
handler = ImageHandler(db_file, table_name="embeddings")

handler.init_embeds_table(f"embeddings")

# Load processor and model
# model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
# processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", do_rescale=True)

# load the model and processor
ckpt = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(ckpt).eval()
processor = AutoProcessor.from_pretrained(ckpt, do_rescale=True)

del model.text_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# filter by ids
# unique_ids = handler.load_existing_ids(where="WHERE status = 200")
all_images = recursively_get_files(image_folder, extensions=['.webp'])
# filtered_images = [img for img in all_images if Path(img).stem in unique_ids]

img_index = 0
batch_size = 90  # Define a batch size
insertion_batch = []  # Temporary list for batch insertion

conn = sqlite3.connect(db_file)
cur = conn.cursor()
cur.execute(f"""SELECT image_id FROM embeddings""")
embedded_ids = [c[0] for c in cur.fetchall()]
embedded_ids = set(embedded_ids)
# filtered_images = [img for img in all_images if Path(img).stem not in embedded_ids]
filtered_images = all_images

with torch.no_grad():
    for img_index in tqdm(range(0, len(filtered_images), batch_size)):
        max_index = min(img_index + batch_size, len(filtered_images))
        total_to_sample = range(img_index, max_index)

        # Embed images
        images = []
        img_names = []
        folder_ids = []
        for i in total_to_sample:
            try:
                images.append(Image.open(filtered_images[i]).convert("RGB"))
                folder_ids.append(filtered_images[i].split('/')[2])
                img_names.append(Path(filtered_images[i]).stem)
            except Exception as e:
                print(e)
                # del filtered_images[i]
        inputs = processor(images=images, padding=True, return_tensors="pt")#.to("cuda:0")
        pixel_values = inputs['pixel_values'].to(device)
        image_embeds = model.get_image_features(pixel_values)

        # Save embeddings to SQLite database
        for i, img_name in enumerate(img_names):
            embedding_blob = image_embeds[i, :].detach().cpu().numpy().tobytes()
            folder_name = folder_ids[i]
            insertion_batch.append((img_name, folder_name, embedding_blob))
        
        if len(insertion_batch) >= batch_size:
            cur.executemany(f"INSERT OR REPLACE INTO embeddings (image_id, folder_id, embedding) VALUES (?, ?, ?)",
                          insertion_batch)
            conn.commit()
            insertion_batch.clear()

# If there are leftover embeddings that were not committed, insert them
if insertion_batch:
    cur.executemany(f"INSERT OR REPLACE INTO embeddings (image_id, folder_id, embedding) VALUES (?, ?, ?)",
                  insertion_batch)
    conn.commit()

conn.close()

# Verify the number of embeddings stored
import sqlite3
conn = sqlite3.connect("embeds.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM embeddings")
count = cursor.fetchone()[0]
print(f"Number of embeddings stored: {count}")
conn.close()

# Verify the shape of the embeddings
import sqlite3
import numpy as np

conn = sqlite3.connect("embeds.db")
cursor = conn.cursor()
cursor.execute("SELECT image_id, embedding FROM embeddings LIMIT 1")
image_id, embedding_blob = cursor.fetchone()
embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)

print(f"Image ID: {image_id}")
print(f"Embedding shape: {embedding_array.shape}")
print(f"Embedding sample: {embedding_array[:5]}")  # First 5 values
conn.close()