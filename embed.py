import openai
from sklearn.metrics.pairwise import cosine_similarity

def get_embeddings(prompt, model="ada-002"):
    completions = openai.Completion.create(
        engine=model,
        prompt=prompt,
        n=1,
        max_tokens=3,
        temperature=0,
        return_prompt=True,
        return_metadata=True
    )

    # Extract the embeddings
    embeddings = completions["choices"][0]["metadata"]["model"]["embedding"]

    return np.array(embeddings)

def build_semantic_index(paragraphs):
    
    semantic_index = []

    for paragraph in paragraphs:
        # Obtain the paragraph embeddings
        embedding = get_embeddings(paragraph)

        # Add the paragraph and its embedding to the index
        semantic_index.append((paragraph, embedding))

    return semantic_index
