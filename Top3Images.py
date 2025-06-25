# === 🏞️ Find top 3 images for each of the top 10 prompts individually ===
from collections import defaultdict

top_prompt_to_scores = defaultdict(list)  # prompt_index -> list of (image_id, score)

emb_index = 0
with torch.no_grad():
    while emb_index < len(embeds):
        upper_index = min(emb_index + 20000, len(embeds))
        batch_embeddings = embeds[emb_index:upper_index]
        batch_ids = embeds_keys[emb_index:upper_index]

        image_embeds = torch.stack(batch_embeddings)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp()
        activations = torch.softmax(logits_per_text, dim=0)  # shape: [num_prompts, num_images]

        for i in topk.indices:  # for each of the top 10 prompts
            prompt_activations = activations[i]  # shape: [num_images]
            for image_id, score in zip(batch_ids, prompt_activations.tolist()):
                top_prompt_to_scores[i.item()].append((image_id, score))

        emb_index += 20000

# === Collate top 3 images per prompt ===
top3_per_prompt = []
for idx in topk.indices:
    prompt_idx = idx.item()
    prompt_text = prompts_in_order[prompt_idx]
    scores = sorted(top_prompt_to_scores[prompt_idx], key=lambda x: x[1], reverse=True)[:3]
    for rank, (image_id, score) in enumerate(scores, start=1):
        top3_per_prompt.append({
            "prompt_index": prompt_idx,
            "prompt_text": prompt_text,
            "rank": rank,
            "image_id": image_id,
            "score": score
        })

top3_per_prompt_df = pd.DataFrame(top3_per_prompt)

# === Export to CSV ===
top3_each_prompt_path = Path("/Users/just/Documents/ERM/Thesis stuff/Output of troubleshooting/top3_images_per_top10_prompt.csv")
top3_each_prompt_path.parent.mkdir(parents=True, exist_ok=True)
top3_per_prompt_df.to_csv(top3_each_prompt_path, index=False)

print("\n📁 Top 3 images per top-10 prompt exported to:")
print(top3_each_prompt_path)