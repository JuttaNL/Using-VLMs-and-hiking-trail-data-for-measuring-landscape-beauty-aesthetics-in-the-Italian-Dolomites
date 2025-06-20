# === 🏞️ Find top 3 images responding to top 10 prompts ===
image_ids = []
image_scores = []
emb_index = 0
with torch.no_grad():
   while emb_index < len(embeds):
       upper_index = min(emb_index + 20000, len(embeds))
       batch_embeddings = embeds[emb_index:upper_index]
       batch_ids = embeds_keys[emb_index:upper_index]


       image_embeds = torch.stack(batch_embeddings)
       image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)


       logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias
       activations = torch.softmax(logits_per_text, dim=0)  # shape: [num_prompts, batch_size]


       top10_activations = activations[topk.indices]        # filter to top 10 prompts
       per_image_score = top10_activations.mean(dim=0)      # mean score per image


       image_ids.extend(batch_ids)
       image_scores.extend(per_image_score.tolist())


       emb_index += 20000


top_image_df = pd.DataFrame({
   "image_id": image_ids,
   "top10_prompt_score": image_scores
}).sort_values(by="top10_prompt_score", ascending=False).head(3)