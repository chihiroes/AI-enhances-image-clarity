import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO
import torch
import requests
import tempfile

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
        try:
            # 创建weights目录
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            status_text = st.empty()

            downloaded = 0
            with open(model_path, 'wb') as f:
                for data in response.iter_content(1024 * 1024):
                    downloaded += len(data)
                    f.write(data)
                    if total_size > 0:
                        percent = int(downloaded / total_size * 100)
                        progress_bar.progress(min(percent, 100))
                        status_text.text(f"下载进度: {percent}%")

            progress_bar.empty()
            status_text.empty()
            st.success("✅ 模型下载完成！")

        except Exception as e:
            st.error(f"模型下载失败: {str(e)}")
            if os.path.exists(model_path):
                os.remove(model_path)
            return False
    return True


# ----------------------------
# 加载模型（带缓存）
# ----------------------------
@st.cache_resource
def load_model(scale=4):
    """加载 Real-ESRGAN 模型"""
    try:
        # 动态导入，避免初始化时出错
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"使用设备: {'GPU 🚀' if device == 'cuda' else 'CPU ⚡'}")

        # 直接指定本地模型路径（无需下载链接）
        os.makedirs("weights", exist_ok=True)
        model_path = f"weights/RealESRGAN_x{scale}plus.pth"

        # 模型路径与下载链接
        model_urls = {
            2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        }

        # 跳过自动下载，直接检查本地模型是否存在
        if not os.path.exists(model_path):
            # 下载模型
            if not download_model_if_needed(model_path, model_urls[scale]):
                return None
            
        # 初始化模型架构
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )

        # 初始化超分模型
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            dni_weight=None,
            device=device,
            tile=400,  # 设置分块大小，避免内存不足
            tile_pad=10,
            pre_pad=0,
            half=(device == 'cuda')  # GPU用半精度
        )
        return upsampler

    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.info("请确保已安装依赖：pip install realesrgan")
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

        # 显示处理进度
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("正在初始化模型...")
        progress_bar.progress(20)

        status_text.text("正在进行超分处理...")
        output, _ = upsampler.enhance(img_bgr, outscale=scale)
        progress_bar.progress(80)

        status_text.text("正在转换结果...")
        # 转回 PIL 图像
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(output_rgb)

        progress_bar.progress(100)
        status_text.text("处理完成！")

        # 清理进度显示
        progress_bar.empty()
        status_text.empty()

        return result_image

    except Exception as e:
        st.error(f"超分处理失败: {str(e)}")
        return None


# ----------------------------
# 图像预处理
# ----------------------------
def preprocess_image(image, max_size=1600):
    """图像预处理，调整过大图像"""
    width, height = image.size

    if max(width, height) > max_size:
        # 计算缩放比例
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)
        st.warning(f"图像尺寸过大，已自动缩放至 {new_width}x{new_height}")

    return image


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

    # 高级设置
    st.sidebar.markdown("---")
    st.sidebar.subheader("高级设置")

    enable_face_enhance = st.sidebar.checkbox(
        "启用人脸增强",
        value=False,
        help="如果图像包含人脸，建议开启此选项"
    )

    max_input_size = st.sidebar.slider(
        "最大输入尺寸",
        min_value=800,
        max_value=3000,
        value=1600,
        help="限制输入图像尺寸以避免内存不足"
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

            # 图像预处理
            original_image = preprocess_image(original_image, max_input_size)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 原始图像")
                st.image(original_image,
                         caption=f"尺寸: {original_image.size[0]}x{original_image.size[1]}",
                         use_container_width=True)

                # 图像信息
                st.info(f"""
                **图像信息:**
                - 格式: {original_image.format or 'Unknown'}
                - 模式: {original_image.mode}
                - 文件大小: {uploaded_file.size // 1024} KB
                - 分辨率: {original_image.size[0]} × {original_image.size[1]}
                """)

            with col2:
                st.subheader("🚀 超分处理")
                if st.button("✨ 开始超分处理", type="primary", use_container_width=True):

                    # 显示处理信息
                    st.write(f"**处理配置:**")
                    st.write(f"- 超分倍数: {scale}倍")
                    st.write(f"- 人脸增强: {'开启' if enable_face_enhance else '关闭'}")
                    st.write(f"- 预计输出尺寸: {original_image.size[0] * scale} × {original_image.size[1] * scale}")

                    result_image = super_resolve(original_image, scale)
                    if result_image:
                        st.success("✅ 超分完成！")
                        st.image(result_image,
                                 caption=f"超分后尺寸: {result_image.size[0]}x{result_image.size[1]} (放大{scale}倍)",
                                 use_container_width=True)

                        # 下载按钮
                        buf = BytesIO()
                        result_image.save(buf, format="PNG", quality=95)
                        st.download_button(
                            label="💾 下载高清图像 (PNG)",
                            data=buf.getvalue(),
                            file_name=f"super_resolution_x{scale}_{original_image.size[0]}x{original_image.size[1]}_to_{result_image.size[0]}x{result_image.size[1]}.png",
                            mime="image/png",
                            use_container_width=True
                        )

                        # 性能统计
                        original_size = uploaded_file.size
                        result_size = len(buf.getvalue())
                        compression_ratio = result_size / original_size if original_size > 0 else 0

                        st.metric("文件大小变化",
                                  f"{original_size // 1024}KB → {result_size // 1024}KB",
                                  f"{compression_ratio:.1f}x")

        except Exception as e:
            st.error(f"图像处理错误: {str(e)}")
            st.info("请尝试上传较小的图像或选择2倍模式")
    else:
        # 使用说明
        st.markdown("""
        ## 📖 使用说明

        1. **上传图像** - 点击上传按钮选择需要处理的图像
        2. **选择倍数** - 在左侧选择 2倍 或 4倍 超分模式  
        3. **开始处理** - 点击"开始超分处理"按钮
        4. **下载结果** - 处理完成后下载高清图像

        ### 💡 使用提示  
        - 首次运行会自动下载模型文件（约100MB）
        - 建议输入图像尺寸在 1600px 以内以获得最佳效果
        - 4倍模式效果更好但需要更多处理时间
        - 如果图像包含人脸，建议开启"人脸增强"选项

        ### 🎯 适用场景
        - 老照片修复和增强
        - 游戏截图和动漫图像放大
        - 低分辨率图像质量提升
        - 文档图像清晰化处理
        """)


if __name__ == "__main__":
    main()