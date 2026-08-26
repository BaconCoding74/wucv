from PIL import Image
from time import perf_counter
from embedding_run import find_item
from pathlib import Path
from dinov2_embedding import DinoV2, EmbeddingStore, ImageRetriever

import os
import streamlit as st
import ir_constants as irc

ITEM_PER_ROW = 5
INPUT_DIR = "image_recognition/datasets/datasets_v002"

st.title("Visualization")
st.set_page_config(page_title="Item Recognition Visualization", layout="wide")



encoder = DinoV2(
    model_name="facebook/dinov2-base", 
    preprocessor_version="1.0",
)

embedding_store = EmbeddingStore(
    cache_path="cache/dinov2_cache.pt", 
    encoder=encoder,
)

stats = embedding_store.sync(
    input_dir=INPUT_DIR, 
    batch_size=32, 
    force_hash_check=False, 
    rebuild_on_mismatch=True,
)

retriever = ImageRetriever(
    encoder=encoder,
    embedding_store=embedding_store,
    device_mode="auto",
    query_batch_size=32,
    reserved_vram=2.0,
    safety_factor=1.2,
)

input_image_paths = [
    path 
    for path in Path(INPUT_DIR).rglob("*") 
    if path.is_file() and path.suffix.lower() in embedding_store.valid_extensions
]

retrieved_results = retriever.retrieve_top_k(
    input_paths=input_image_paths,
)

for input_path, results in zip(input_image_paths, retrieved_results):
    print(f"Input image: {input_path}")

    for result in results:
        color_code = "\033[92m"

        # If query and retrieved image are from different folders, print in red
        if input_path.parent.name != Path(result["retrieved_path"]).parent.name:
            color_code = "\033[31m"

        print(f"{color_code}\tRetrieved image: {result['retrieved_path']}, Score: {result['score']:.4f}\033[0m")

for img, top in results.items():
    container = st.container(border=True)
    with container:
        st.subheader(f"Query: {img}")
        st.image(f"{irc.TEST_INPUTS_PATH}/{img}", caption="Query Image")

        for i in range(0, len(top), ITEM_PER_ROW):
            columns = st.columns(ITEM_PER_ROW)
            row_items = top[i:i + ITEM_PER_ROW]

            for j, (name, score) in enumerate(row_items):
                with columns[j]:
                    st.write(f"Top {i + j + 1}: {name}")
                    st.write(f"Score: {score:.4f}")
                    st.image(f"{irc.TEST_REFERENCES_PATH}/{name}/{os.listdir(f'{irc.TEST_REFERENCES_PATH}/{name}')[0]}", caption=f"Reference Image: {name}")