import tensorflow as tf
import cv2
import numpy as np
import os
import streamlit as st
import io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")
st.title("Исследование пучков заряженных частиц")
uploaded_file = st.file_uploader("Загрузите PGM или NPY файл", type=["pgm", "npy"])

#Загрузка модели
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:\\vkr\\model.keras", compile=False)

model = load_model()

#Функция Гаусса
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

#Интегрирование по двум осям и задание масштаба
def calculate_profiles(beam_crop, k=1.0):
    col_sums = np.sum(beam_crop, axis=0)
    row_sums = np.sum(beam_crop, axis=1)
    scaled_coords = np.arange(len(col_sums)) * k
    return col_sums, row_sums, scaled_coords

#Построение графика с аппроксимацией
def plot_single_profile(ax, data, title, xlabel, ylabel, k=1.0, time_unit="нс"):
    x_data = np.arange(len(data)) * k
    y_data = data

    try:
        initial_guess = [max(y_data), len(y_data)/2 * k, len(y_data)/4 * k]
        params, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
        A_fit, mu_fit, sigma_fit = params

        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = gaussian(x_fit, A_fit, mu_fit, sigma_fit)

        ax.plot(x_data, y_data, label='Исходные данные')
        ax.plot(x_fit, y_fit, label='Аппроксимация', linestyle='--', color='red')
        ax.legend(loc='upper left')

        fit_text = f"Функция: {A_fit:.2f}·exp(-(x-{mu_fit:.2f})²/(2·{sigma_fit:.2f}²))"
        ax.set_xlabel(f"{xlabel}, {time_unit}\n\n{fit_text}", fontsize=10)
        
    except RuntimeError:
        ax.plot(x_data, y_data, label='Исходные данные')
        ax.text(0.5, 0.5, "Аппроксимация не удалась",
                transform=ax.transAxes,
                fontsize=12, color='red',
                ha='center', va='center')

    ax.set_title(title)
    ax.set_ylabel("Интенсивность")

#Функция построения профилей
def plot_profiles(row_sums, col_sums, beam_idx, k=1.0, time_unit="нс"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_single_profile(ax1, row_sums,
                       f'Вертикальный профиль (пучок #{beam_idx})',
                       'Время', 'Интенсивность', k, time_unit)
    plot_single_profile(ax2, col_sums,
                       f'Горизонтальный профиль (пучок #{beam_idx})',
                       'Время', 'Интенсивность', k, time_unit)
    plt.tight_layout()
    return fig

#Сохранение графиков
def save_profile_plots(row_sums, col_sums, beam_idx, k=1.0, time_unit="нс"):
    buf_vertical = io.BytesIO()
    buf_horizontal = io.BytesIO()

    fig_vertical = plt.figure(figsize=(8, 5))
    ax_vertical = fig_vertical.add_subplot(111)
    plot_single_profile(ax_vertical, row_sums,
                       f'Вертикальный профиль (пучок #{beam_idx})',
                       'Время', 'Интенсивность', k, time_unit)
    fig_vertical.savefig(buf_vertical, format='png', bbox_inches='tight')
    plt.close(fig_vertical)

    fig_horizontal = plt.figure(figsize=(8, 5))
    ax_horizontal = fig_horizontal.add_subplot(111)
    plot_single_profile(ax_horizontal, col_sums,
                       f'Горизонтальный профиль (пучок #{beam_idx})',
                       'Время', 'Интенсивность', k, time_unit)
    fig_horizontal.savefig(buf_horizontal, format='png', bbox_inches='tight')
    plt.close(fig_horizontal)

    buf_vertical.seek(0)
    buf_horizontal.seek(0)

    return buf_vertical, buf_horizontal

#Загрузка изображения
def load_image(img_path):
    file_extension = os.path.splitext(img_path.name)[1].lower()

    if file_extension == '.pgm':
        original_img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            st.error(f"Ошибка загрузки PGM файла: {img_path.name}")
            return None, None

        display_img = ((original_img / original_img.max()) * 255).astype(np.uint8)

    elif file_extension == '.npy':
        try:
            original_img = np.load(io.BytesIO(img_path.read()))
            if len(original_img.shape) != 2:
                st.error(f"Ошибка: NPY файл должен содержать двумерный массив. Форма: {original_img.shape}")
                return None, None

            min_val = np.min(original_img)
            max_val = np.max(original_img)
            if not (0 <= min_val and max_val <= 255):
                st.error(f"Ошибка: Диапазон значений вне [0,255] (min={min_val}, max={max_val})")
                return None, None

            original_img = original_img.astype(np.uint8)
            display_img = original_img
        except Exception as e:
            st.error(f"Ошибка загрузки NPY файла: {e}")
            return None, None
    else:
        st.error(f"Неподдерживаемый формат: {file_extension}")
        return None, None

    return original_img, display_img

#Боковая панель управления
def get_sidebar_controls(display_img):
    with st.sidebar:
        st.subheader("Исходное изображение")
        st.image(display_img, caption="Оригинал", use_container_width=True, channels='GRAY')

        resized_img = cv2.resize(display_img, (256, 256))
        normalized_img = resized_img / 255.0
        tensor_input = np.expand_dims(normalized_img, axis=0)
        prediction = np.argmax(model.predict(tensor_input))
        st.write(f"Модель предсказала: {prediction}")

        col1, col2 = st.columns([3, 2])
        
        with col1:
            k = st.number_input(
                "Коэффициент масштабирования (k)",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.1,
                format="%.2f"
            )
        
        with col2:
            time_unit = st.selectbox(
                "Единицы времени",
                options=["нс", "пс"],
                index=0  # По умолчанию выбраны наносекунды
            )

    return prediction, k, time_unit

#Обнаружение пучков
def detect_beams(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 300 < area < image.size * 0.5:
            x, y, w, h = cv2.boundingRect(contour)
            beam_crop = image[y:y+h, x:x+w]
            beam_display = ((beam_crop / beam_crop.max()) * 255).astype(np.uint8)
            detected_objects.append((x, y, w, h, beam_crop, beam_display))

    return detected_objects

#Отображение области пучка
def show_beam_results(beam_data, k, time_unit):
    valid_idx = 1
    for x, y, w, h, beam_crop, beam_display in beam_data:
        col_sums, row_sums, scaled_coords = calculate_profiles(beam_crop, k)

        st.image(beam_display, caption=f"Пучок #{valid_idx}", width=400, channels='GRAY')

        st.markdown(f"""
            **Координаты:**  
            X: `{x}`  
            Y: `{y}`  
            Ширина: `{w}`  
            Высота: `{h}`
        """)

        fig = plot_profiles(row_sums, col_sums, valid_idx, k, time_unit)
        st.pyplot(fig)

        buf_vert, buf_horiz = save_profile_plots(row_sums, col_sums, valid_idx, k, time_unit)

        st.download_button(
            label=f"Скачать вертикальный профиль #{valid_idx}",
            data=buf_vert,
            file_name=f"vertical_profile_beam_{valid_idx}.png",
            mime="image/png"
        )

        st.download_button(
            label=f"Скачать горизонтальный профиль #{valid_idx}",
            data=buf_horiz,
            file_name=f"horizontal_profile_beam_{valid_idx}.png",
            mime="image/png"
        )

        valid_idx += 1

#Основная функция обработки
def selectArea(img_path, model):
    original_img, display_img = load_image(img_path)
    if original_img is None:
        return []

    prediction, k, time_unit = get_sidebar_controls(display_img)

    if prediction != 1:
        st.warning("Пучок не обнаружен")
        return []

    beams = detect_beams(original_img)

    if not beams:
        st.info("Пучки не найдены")
        return []

    show_beam_results(beams, k, time_unit)

    return [(x, y, w, h) for x, y, w, h, _, _ in beams]

if uploaded_file:
    selectArea(uploaded_file, model)