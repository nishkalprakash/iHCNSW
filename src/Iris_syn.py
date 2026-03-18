from knns.hnsw import HNSW


import os
import random
import numpy as np


def load_iris_txt_templates(root_folder, seed=42, pick='random', normalize=True,
                            num_index_per_subject=7):
    """
    Dataset structure:
    root_folder/
        9978/
            1_template.txt
            1_mask.txt
            ...
            10_template.txt
            10_mask.txt

    For each subject:
    - pick 3 pairs for indexing
    - remaining pairs go to query pool
    - finally sample 10000 queries randomly from the full query pool

    Assumption:
    - template.txt contains binary iris code bits
    - mask.txt contains binary validity bits
    - mask == 1 means valid
    - mask == 0 means invalid
    - final vector = template & mask
    """

    random.seed(seed)
    np.random.seed(seed)

    if not os.path.isdir(root_folder):
        raise ValueError(f"Root folder not found: {root_folder}")

    subject_folders = sorted([
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ])

    if not subject_folders:
        raise ValueError(f"No subject folders found in {root_folder}")

    def read_txt_bits(path):
        with open(path, "r") as f:
            content = f.read()

        bits = [int(ch) for ch in content if ch in "01"]

        if len(bits) == 0:
            raise ValueError(f"No binary digits found in {path}")

        return np.array(bits, dtype=np.uint8)

    def pair_to_vector(template_path, mask_path):
        template = read_txt_bits(template_path)
        mask = read_txt_bits(mask_path)

        if template.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"template: {template.shape} from {template_path}\n"
                f"mask: {mask.shape} from {mask_path}"
            )

        vec = (template & mask).astype(np.float32)

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec

    index_vectors = []
    index_labels = []
    query_pool_vectors = []
    query_pool_labels = []

    for subject in subject_folders:
        subject_path = os.path.join(root_folder, subject)

        pairs = []
        for i in range(1, 11):
            template_file = os.path.join(subject_path, f"{i}_template.txt")
            mask_file = os.path.join(subject_path, f"{i}_mask.txt")

            if os.path.exists(template_file) and os.path.exists(mask_file):
                pairs.append((i, template_file, mask_file))

        if len(pairs) == 0:
            print(f"⚠️ No valid txt pairs found for subject {subject}, skipping.")
            continue

        if len(pairs) < num_index_per_subject:
            print(f"⚠️ Subject {subject} has only {len(pairs)} pairs, skipping.")
            continue

        if pick == 'random':
            random.shuffle(pairs)
        elif pick == 'first':
            pairs = sorted(pairs, key=lambda x: x[0])
        else:
            raise ValueError("pick must be 'random' or 'first'")

        index_pairs = pairs[:num_index_per_subject]
        query_pairs = pairs[num_index_per_subject:]

        index_count = 0
        query_count = 0

        for idx_num, idx_template, idx_mask in index_pairs:
            try:
                idx_vec = pair_to_vector(idx_template, idx_mask)
                index_vectors.append(idx_vec)
                index_labels.append(subject)
                index_count += 1
            except Exception as e:
                print(f"⚠️ Failed index pair for subject {subject}, image {idx_num}: {e}")

        for q_num, q_template, q_mask in query_pairs:
            try:
                q_vec = pair_to_vector(q_template, q_mask)
                query_pool_vectors.append(q_vec)
                query_pool_labels.append(subject)
                query_count += 1
            except Exception as e:
                print(f"⚠️ Failed query pair for subject {subject}, image {q_num}: {e}")

        print(f"{subject}: index={index_count}, query_pool={query_count}")

    if len(index_vectors) == 0:
        raise ValueError("No index vectors created. Check txt file contents.")

    if len(query_pool_vectors) == 0:
        raise ValueError("No query pool vectors created.")

    # Convert index data
    index_vectors = np.vstack(index_vectors).astype(np.float32)
    index_labels = np.array(index_labels)

    query_vectors = query_pool_vectors
    query_labels = query_pool_labels

    return index_vectors, index_labels, query_vectors, query_labels
# i will give the path for the dataset , use that here to test the implementation of the HNSW with the iris dataset
if __name__ == "__main__":    

    # Load the iris dataset
    # iris = load_iris()
    root_folder = "/home/nishkal/sg/iris_indexing/datasets/iris_syn"
    index_vectors, index_labels, query_vectors, query_labels = load_iris_txt_templates(
        root_folder=root_folder,
        seed=42,
        pick='random'
    )
    

    # Create an instance of the HNSW class
    hnsw = HNSW(m=32, ef_construction=900, ef=100)
    # Insert the data into the HNSW graph
    hnsw.insert(index_vectors)
    # Now you can perform searches using hnsw.search(query_vector, k) where query_vector is a vector from query_vectors and k is the number of nearest neighbors you want to retrieve. You can also evaluate the search results against the query_labels to check for accuracy.

    # do the search for all the queries and check if it retrieves the correct subject label in the top k results and give me hit rate like in those queries how many retrieved correct
    k = 1
    correct_hits = 0
    total_queries = len(query_vectors) 
    for i in range(total_queries):
        query_vec = query_vectors[i]
        query_label = query_labels[i]
        results = hnsw.search(query_vec, k=k)
        retrieved_labels = [index_labels[idx] for idx, dist in results]
        if query_label in retrieved_labels:
            correct_hits += 1

    hit_rate = correct_hits / total_queries if total_queries > 0 else 0
    print(f"Hit rate: {hit_rate:.2f}")