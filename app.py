
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import cv2
import io

# Define all effects
def apply_effect(img, effect):
    img_np = np.array(img)

    if effect == "Original":
        return img

    elif effect == "Black & White":
        return img.convert("L").convert("RGB")

    elif effect == "Sepia":
        sepia = np.array(img.convert("RGB"))
        tr = [0.393, 0.769, 0.189]
        tg = [0.349, 0.686, 0.168]
        tb = [0.272, 0.534, 0.131]
        r, g, b = sepia[:,:,0], sepia[:,:,1], sepia[:,:,2]
        sepia[:,:,0] = np.clip(r*tr[0] + g*tr[1] + b*tr[2], 0, 255)
        sepia[:,:,1] = np.clip(r*tg[0] + g*tg[1] + b*tg[2], 0, 255)
        sepia[:,:,2] = np.clip(r*tb[0] + g*tb[1] + b*tb[2], 0, 255)
        return Image.fromarray(sepia.astype(np.uint8))

    elif effect == "Posterize":
        return ImageOps.posterize(img, bits=3)

    elif effect == "Cartoon Edges":
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 5), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        color = cv2.bilateralFilter(np.array(img), 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return Image.fromarray(cartoon)

    elif effect == "Duotone":
        gray = img.convert("L")
        return ImageOps.colorize(gray, black="navy", white="gold")

    elif effect == "Soft Pastel":
        pastel = ImageEnhance.Color(img).enhance(0.5)
        pastel = ImageEnhance.Brightness(pastel).enhance(1.1)
        blur = cv2.GaussianBlur(np.array(pastel), (3,3), 1)
        return Image.fromarray(blur)

    elif effect == "Dreamy Glow":
        blur = img.filter(ImageFilter.GaussianBlur(radius=10))
        return Image.blend(img, blur, alpha=0.4)

    elif effect == "Watercolor":
        return Image.fromarray(cv2.bilateralFilter(np.array(img), 9, 75, 75))

    elif effect == "Focus + Desaturated BG":
        img_np = np.array(img)
        rows, cols = img_np.shape[:2]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (cols//2, rows//2), min(cols, rows)//3, 255, -1)
        mask = cv2.GaussianBlur(mask, (151, 151), 50) / 255.0
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        blended = (img_np * mask[..., None] + gray_rgb * (1 - mask[..., None])).astype(np.uint8)
        return Image.fromarray(blended)

    elif effect == "Cel-shaded Look":
        flat = ImageOps.posterize(img, bits=4)
        edges = img.filter(ImageFilter.FIND_EDGES).convert("L").point(lambda x: 255 if x > 40 else 0)
        outline = ImageOps.invert(edges).convert("RGB")
        return Image.blend(flat, outline, alpha=0.3)

    elif effect == "Pencil Sketch":
        gray = img.convert("L")
        inverted = ImageOps.invert(gray)
        blurred = inverted.filter(ImageFilter.GaussianBlur(radius=10))
        blend = Image.blend(gray, blurred, alpha=0.5)
        return ImageOps.invert(blend)

    elif effect == "Glitch RGB Shift":
        img_np = np.array(img)
        glitch = img_np.copy()
        glitch[:,:,0] = np.roll(glitch[:,:,0], 5, axis=1)
        glitch[:,:,2] = np.roll(glitch[:,:,2], -5, axis=0)
        glitch[::2] = glitch[::2] // 2
        return Image.fromarray(glitch)

    elif effect == "Foggy Ambient":
        fog = cv2.GaussianBlur(np.array(img), (51, 51), 30)
        foggy = cv2.addWeighted(np.array(img), 0.6, fog, 0.4, 0)
        return Image.fromarray(foggy)

    elif effect == "Color Warp (Fantasy)":
        arr = np.array(img).astype(np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        arr[:,:,0] = np.clip(r * 0.8 + g * 0.1 + 20, 0, 255)
        arr[:,:,1] = np.clip(g * 0.5 + b * 0.2, 0, 255)
        arr[:,:,2] = np.clip(b * 1.2 + r * 0.1 + 30, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    elif effect == "Canvas Texture Overlay":
        base_np = np.array(img)
        canvas = np.zeros((img.height, img.width), dtype=np.uint8)
        for i in range(0, img.height, 2):
            canvas[i] = np.random.randint(190, 255, size=img.width)
        for j in range(0, img.width, 2):
            canvas[:, j] = np.minimum(canvas[:, j], np.random.randint(190, 255, size=img.height))
        canvas = cv2.GaussianBlur(canvas, (3, 3), 1)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(base_np, 0.95, canvas_rgb, 0.05, 0)
        return Image.fromarray(blended)

    elif effect == "Sharp Outline + Paint":
        flat = ImageOps.posterize(img, bits=5)
        edges = img.filter(ImageFilter.FIND_EDGES).convert("L").point(lambda x: 255 if x > 30 else 0)
        return Image.blend(flat, ImageOps.invert(edges).convert("RGB"), alpha=0.2)

    elif effect == "Soft Focus Portrait":
        blur = cv2.GaussianBlur(np.array(img), (0, 0), sigmaX=3)
        return Image.fromarray(cv2.addWeighted(np.array(img), 0.75, blur, 0.25, 0))

    elif effect == "Perspective Warp":
        img_np = np.array(img)
        rows, cols = img_np.shape[:2]
        src = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
        dst = np.float32([[cols*0.1,rows*0.1],[cols*0.9,rows*0.05],
                          [cols*0.15,rows*0.9],[cols*0.85,rows*0.95]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img_np, matrix, (cols, rows))
        return Image.fromarray(warp)

    elif effect == "Posterize + Blur":
        poster = ImageOps.posterize(img, bits=4)
        return poster.filter(ImageFilter.MedianFilter(size=3))

    else:
        return img

# List of all effects
effects = [
    "Original", "Black & White", "Sepia", "Posterize", "Cartoon Edges", "Duotone",
    "Soft Pastel", "Dreamy Glow", "Watercolor", "Focus + Desaturated BG",
    "Cel-shaded Look", "Pencil Sketch", "Glitch RGB Shift", "Foggy Ambient",
    "Color Warp (Fantasy)", "Canvas Texture Overlay", "Sharp Outline + Paint",
    "Soft Focus Portrait", "Perspective Warp", "Posterize + Blur"
]

# --- Streamlit App ---
st.set_page_config(page_title="Ghibli Art Lab – Full Style Preview", layout="wide")
st.title("🖼️ Ghibli Art Lab – Preview All 20 Styles")
st.caption("Upload a photo and preview all effects in one view. Click to download any version.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("🎨 Style Previews")

    cols = st.columns(3)
    for i, effect in enumerate(effects):
        with cols[i % 3]:
            transformed = apply_effect(image, effect)
            st.image(transformed, caption=effect, use_column_width=True)

            buf = io.BytesIO()
            transformed.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="💾 Download",
                data=byte_im,
                file_name=f"{effect.lower().replace(' ', '_')}.png",
                mime="image/png",
                key=f"dl_{i}"
            )
