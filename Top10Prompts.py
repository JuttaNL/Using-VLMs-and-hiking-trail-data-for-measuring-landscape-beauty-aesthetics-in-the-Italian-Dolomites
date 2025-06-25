# === Top Prompt Influence Analysis ===
avg_activations = total_activations / num_images_total
topk = torch.topk(avg_activations, k=10)


print("\n🌟 Top 10 Prompts by Average Influence Across All Images:")
for i, idx in enumerate(topk.indices):
   print(f"{i+1}. {prompts_in_order[idx]} (Avg. Activation: {avg_activations[idx]:.4f})")