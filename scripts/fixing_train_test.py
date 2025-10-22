import numpy as np
import os

# === CONFIG ===
old_train_path = "../old_embeddings_files/resnet_embeddings_train.npz"
old_test_path = "../old_embeddings_files/resnet_embeddings_test.npz"
new_train_list_path = "../dataset/cropped/cropped_train.txt"
new_test_list_path = "../dataset/cropped/cropped_test.txt"
output_train_path = "../embeddings_files/resnet_embeddings__train.npz"
output_test_path = "../embeddings_files/resnet_embeddings__test.npz"

# === LOAD OLD EMBEDDINGS ===
def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    image_paths = data["image_paths"]
    return embeddings, image_paths

train_embeddings, train_paths = load_embeddings(old_train_path)
test_embeddings, test_paths = load_embeddings(old_test_path)

# Combine all into one lookup
all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
all_paths = np.concatenate([train_paths, test_paths], axis=0)
path_to_embedding = {p: e for p, e in zip(all_paths, all_embeddings)}

# === LOAD NEW SPLIT LISTS ===
def load_path_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

new_train_paths = load_path_list(new_train_list_path)
new_test_paths = load_path_list(new_test_list_path)

# === REASSIGN EMBEDDINGS ===
new_train_embs, new_train_imgs = [], []
new_test_embs, new_test_imgs = [], []
missing = []

for p in new_train_paths:
    if p in path_to_embedding:
        new_train_embs.append(path_to_embedding[p])
        new_train_imgs.append(p)
    else:
        missing.append(p)

for p in new_test_paths:
    if p in path_to_embedding:
        new_test_embs.append(path_to_embedding[p])
        new_test_imgs.append(p)
    else:
        missing.append(p)

if missing:
    print(f"⚠️ Warning: {len(missing)} image paths not found in the old embeddings.")
    for m in missing[:10]:
        print("   -", m)
    if len(missing) > 10:
        print("   ... (truncated)")

# === CONVERT TO ARRAYS ===
new_train_embs = np.vstack(new_train_embs)
new_test_embs = np.vstack(new_test_embs)
new_train_imgs = np.array(new_train_imgs)
new_test_imgs = np.array(new_test_imgs)

# === SAVE NEW FILES (same format as original extraction) ===
np.savez(output_train_path, embeddings=new_train_embs, image_paths=new_train_imgs)
np.savez(output_test_path, embeddings=new_test_embs, image_paths=new_test_imgs)

print(f"✅ Saved new training embeddings to: {output_train_path}")
print(f"✅ Saved new test embeddings to: {output_test_path}")

# === LOAD OLD EMBEDDINGS ===
def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    image_paths = data["image_paths"]
    return embeddings, image_paths

train_embeddings, train_paths = load_embeddings(old_train_path)
test_embeddings, test_paths = load_embeddings(old_test_path)

# Combine everything into one big pool
all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
all_paths = np.concatenate([train_paths, test_paths], axis=0)

# === LOAD NEW SPLIT PATHS ===
def load_path_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

new_train_paths = set(load_path_list(new_train_list_path))
new_test_paths = set(load_path_list(new_test_list_path))

# === BUILD A LOOKUP FOR EMBEDDINGS ===
path_to_embedding = {p: emb for p, emb in zip(all_paths, all_embeddings)}

# === REASSIGN ===
new_train_embs, new_train_imgs = [], []
new_test_embs, new_test_imgs = [], []

missing = []

for p in new_train_paths:
    if p in path_to_embedding:
        new_train_embs.append(path_to_embedding[p])
        new_train_imgs.append(p)
    else:
        missing.append(p)

for p in new_test_paths:
    if p in path_to_embedding:
        new_test_embs.append(path_to_embedding[p])
        new_test_imgs.append(p)
    else:
        missing.append(p)

if missing:
    print(f"⚠️ Warning: {len(missing)} image paths not found in the old embeddings.")
    for m in missing[:10]:
        print("   -", m)
    if len(missing) > 10:
        print("   ... (truncated)")

# === SAVE NEW FILES ===
np.savez_compressed(
    output_train_path,
    embeddings=np.array(new_train_embs),
    image_paths=np.array(new_train_imgs)
)

np.savez_compressed(
    output_test_path,
    embeddings=np.array(new_test_embs),
    image_paths=np.array(new_test_imgs)
)

print(f"✅ Saved new training embeddings to: {output_train_path}")
print(f"✅ Saved new test embeddings to: {output_test_path}")
