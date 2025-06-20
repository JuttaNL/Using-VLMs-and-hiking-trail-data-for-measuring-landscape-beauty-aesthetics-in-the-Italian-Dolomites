from scipy.stats import kendalltau

# ChatGPT and DeepSeek scenicness scores from the image
chatgpt_scores = ["insert list of ChatGPT scores"]

deepseek_scores = ["Inster list of DeepSeek scores"]

# Calculate Kendall's tau
tau, _ = kendalltau(chatgpt_scores, deepseek_scores)
print(f"Kendall’s tau: {tau:.3f}")