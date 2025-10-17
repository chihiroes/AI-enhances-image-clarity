import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO
import torch
import requests

# 页面配置
st.set_page_config(
    page_title="AI图像超分工具",
    page_icon="🖼️",
    layout="wide"
)

st.title("🎨 AI图像超分辨率工具")
st.markdown("将低分辨率图像转换为高清图像，支持2倍/4倍放大")


# ----------------------------
# 工具函数：下载模型文件（仅第一次）
# ----------------------------
def download_model_if_needed(model_path: str, url: str):
    """自动下载模型文件到本地"""
    if not os.path.exists(model_path):
        st.warning(f"首次运行，正在下载模型文件，请稍等（约100MB）...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(model_path, 'wb') as f:
            for data in response.iter_content(1024 * 1024):
                downloaded += len(data)
                f.write(data)
                percent = int(downloaded / total_size * 100)
                st.progress(min(percent, 100))
        st.success("✅ 模型下载完成！")


# ----------------------------
# 加载模型（带缓存）
# ----------------------------
@st.cache_resource
def load_model(scale=4):
    """加载 Real-ESRGAN 模型"""
    try:
        from realesrgan import RealESRGANer

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"使用设备: {'GPU 🚀' if device == 'cuda' else 'CPU ⚡'}")

        # 模型路径与下载链接
        model_urls = {
            2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        }
        model_path = f"./RealESRGAN_x{scale}plus.pth"
        download_model_if_needed(model_path, model_urls[scale])

        # 初始化超分模型
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            dni_weight=None,
            device=device,
            tile=0,  # 不分块
            tile_pad=10,
            pre_pad=0,
            half=(device == 'cuda')  # GPU用半精度
        )
        return upsampler

    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None


# ----------------------------
# 执行超分辨率处理
# ----------------------------
def super_resolve(image, scale=4):
    """执行 AI 超分辨率"""
    try:
        upsampler = load_model(scale)
        if upsampler is None:
            return None

        # 转换图像为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner(f"AI 正在进行 {scale} 倍超分处理中..."):
            output, _ = upsampler.enhance(img_bgr, outscale=scale)

        # 转回 PIL 图像
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)

    except Exception as e:
        st.error(f"超分处理失败: {str(e)}")
        return None


# ----------------------------
# 主界面逻辑
# ----------------------------
def main():
    # 侧边栏配置
    st.sidebar.title("⚙️ 参数设置")

    # 选择倍数
    scale = st.sidebar.radio(
        "选择超分倍数",
        [2, 4],
        index=1,
        help="2倍：速度较快；4倍：效果更好"
    )

    # 上传文件
    uploaded_file = st.file_uploader(
        "📤 上传低分辨率图像",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="支持 JPG, PNG, BMP, WEBP 格式"
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 原始图像")
                st.image(original_image,
                         caption=f"尺寸: {original_image.size[0]}x{original_image.size[1]}",
                         use_container_width=True)
                st.info(f"""
                **图像信息:**
                - 格式: {original_image.format or 'Unknown'}
                - 模式: {original_image.mode}
                - 文件大小: {uploaded_file.size // 1024} KB
                """)

            with col2:
                st.subheader("🚀 超分处理")
                if st.button("✨ 开始超分处理", type="primary"):
                    max_size = 2000
                    if max(original_image.size) > max_size:
                        st.warning("图像尺寸较大，处理可能需要较长时间")

                    result_image = super_resolve(original_image, scale)
                    if result_image:
                        st.success("✅ 超分完成！")
                        st.image(result_image,
                                 caption=f"超分后尺寸: {result_image.size[0]}x{result_image.size[1]}",
                                 use_container_width=True)

                        # 下载按钮
                        buf = BytesIO()
                        result_image.save(buf, format="PNG")
                        st.download_button(
                            label="💾 下载高清图像",
                            data=buf.getvalue(),
                            file_name=f"super_resolution_x{scale}.png",
                            mime="image/png"
                        )
        except Exception as e:
            st.error(f"图像处理错误: {str(e)}")
    else:
        st.markdown("""
        ## 📖 使用说明
        1. 点击上传按钮选择图像  
        2. 在左侧选择 2倍 或 4倍 模式  
        3. 点击 “开始超分处理”  
        4. 下载高清图像结果  

        💡 **提示**  
        - 首次运行会自动下载模型（约100MB）  
        - 适用于：老照片修复、截图放大、模糊图片增强  
        """)


if __name__ == "__main__":
    main()
