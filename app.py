import streamlit as st
from transformers import pipeline,AutoProcessor
from PIL import Image, ImageDraw
import numpy as np

# Streamlit UI の設定
st.title("🚀 物体検出アプリ（Hugging Face Transformers）")
st.write("アップロードした画像から物体を検出します")

# 物体検出モデルの設定（Hugging Face Transformers）
checkpoint = "PekingU/rtdetr_v2_r50vd"  # ここに使用するモデル名を指定
image_processor = AutoProcessor.from_pretrained(checkpoint, use_fast=True)
pipe = pipeline("object-detection", model=checkpoint,image_processor=image_processor)

# ユーザーが画像をアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# 閾値を調整できるスライダー
threshold = st.slider("検出の閾値（0.0〜1.0）", 0.0, 1.0, 0.3, 0.05)

if uploaded_file:
    # 画像を開く
    image = Image.open(uploaded_file).convert("RGB")
    
    # Streamlit で元画像を表示
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    # 物体検出を実行
    with st.spinner("物体検出中..."):
        results = pipe(image, threshold=threshold)

    # 結果を描画
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # カラーをランダム生成
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(results), 3))

    for i, result in enumerate(results):
        box = result["box"]
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        color = tuple([int(x) for x in COLORS[i]])

        # 矩形を描画
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=3)
        
        # ラベルを描画
        draw.text((xmin, ymin), text=f"{result['label']} ({result['score']:.2f})", fill=color)

    # 検出結果を表示
    st.image(annotated_image, caption="物体検出結果", use_container_width=True)
    st.success("物体検出完了！")
