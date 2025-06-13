import tensorflow as tf
import cv2
import numpy as np
import os
import streamlit as st
import io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from zipfile import ZipFile
import tempfile
import shutil
from PIL import Image

st.set_page_config(layout="wide")
st.title("Исследование пучков заряженных частиц")

#Загрузка модели
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:\\python\\PREDFINAL\\model.keras", compile=False)

model = load_model()

#Загрузка файла
def load_image(file_content, file_extension):
    if file_extension == '.pgm':
        try:
            content = file_content.decode('utf-8').split('\n')
            content = [line.strip() for line in content if line.strip() and not line.startswith('#')]

            if content[0] != 'P2':
                st.error("Поддерживается только PGM формат P2 (текстовый)")
                return None, None

            width, height = map(int, content[1].split())
            max_val = int(content[2])

            pixels = []
            for line in content[3:]:
                pixels.extend(map(int, line.split()))

            dtype = np.uint16 if max_val > 255 else np.uint8
            original_img = np.array(pixels, dtype=dtype).reshape((height, width))
            display_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        except Exception as e:
            st.error(f"Ошибка загрузки PGM файла: {e}")
            return None, None

    elif file_extension == '.npy':
        try:
            original_img = np.load(io.BytesIO(file_content))
            if len(original_img.shape) != 2:
                st.error("NPY файл должен содержать двумерный массив")
                return None, None

            display_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        except Exception as e:
            st.error(f"Ошибка загрузки NPY файла: {e}")
            return None, None

    else:
        st.error(f"Неподдерживаемый формат: {file_extension}")
        return None, None

    return original_img, display_img

# Загрузка одиночного файла
def handle_single_file():
    uploaded_file = st.file_uploader("Загрузите PGM или NPY файл",
                                     type=["pgm", "npy"],
                                     accept_multiple_files=False,
                                     key="single_file_uploader")
    if uploaded_file is not None:
        with st.spinner("Обработка файла..."):
            beams = process_single_file(uploaded_file.getvalue(), uploaded_file.name)

# Загрузка zip-архива
def handle_zip_archive():
    uploaded_zip = st.file_uploader("Загрузите ZIP архив с изображениями",
                                    type=["zip"],
                                    accept_multiple_files=False,
                                    key="zip_uploader")

    if uploaded_zip is not None:
        temp_dir = tempfile.mkdtemp()
        try:
            with st.spinner("Обработка архива..."):
                output_zip_path, processed_files = process_zip_archive(uploaded_zip, temp_dir)

                if processed_files:
                    st.success(f"Успешно обработано {len(processed_files)} файлов")

                    original_archive_name = os.path.splitext(uploaded_zip.name)[0]
                    result_archive_name = f"{original_archive_name}_results.zip"

                    offer_zip_download(output_zip_path, result_archive_name)

                    with st.expander("Показать список обработанных файлов"):
                        st.write(processed_files)
                else:
                    st.warning("Не удалось обработать ни одного файла с пучками")

        except Exception as e:
            st.error(f"Произошла ошибка при обработке архива: {str(e)}")
            st.exception(e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            st.cache_resource.clear()

#Предобработка файла для передачи в НС, ответ модели НС
def preprocess_and_predict(image):
    img_max = float(image.max())
    normalized_img = image / img_max if img_max > 0 else image
    resized_img = cv2.resize(normalized_img, (256, 256))
    tensor_input = np.expand_dims(resized_img, axis=0)
    prediction_prob = model.predict(tensor_input)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0
    return prediction

#Отображение загруженного файла и ответа модели НС в боковой панели
def show_image_and_prediction(display_img, original_img):
    with st.sidebar:
        st.subheader("Исходное изображение")
        st.image(display_img, caption="Оригинал", use_container_width=True, channels='GRAY')

        prediction = preprocess_and_predict(original_img)
        st.write(f"Модель предсказала: {prediction}")

    return prediction

#Отображение поля ввода коэффициента масштабирования и выбора единиц измерения времени в боковой панели
def get_processing_params():
    with st.sidebar:
        st.subheader("Параметры обработки")

        col1, col2 = st.columns(2)
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
                ["нс", "пс"],
                index=0
            )

    return k, time_unit

#Обнаружение пучков
def detect_beams(image):
    if image.dtype == np.uint16:
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        image_8bit = image

    blurred = cv2.GaussianBlur(image_8bit, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 300 < area < image.size * 0.5:
            x, y, w, h = cv2.boundingRect(contour)
            beam_crop = image[y:y + h, x:x + w]
            beam_display = ((beam_crop / beam_crop.max()) * 255).astype(np.uint8)
            detected_objects.append((x, y, w, h, beam_crop, beam_display))

    return detected_objects

#Функция Гаусса
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

#Интегрирование по двум осям с заданным масштабом
def calculate_profiles(beam_crop, k=1.0):
    col_sums = np.sum(beam_crop, axis=0)
    row_sums = np.sum(beam_crop, axis=1)

    x_horizontal = np.arange(beam_crop.shape[1]) * k
    x_vertical = np.arange(beam_crop.shape[0]) * k

    return col_sums, row_sums, x_horizontal, x_vertical

#Построение графика с аппроксимацией
def plot_profile(ax, x_data, y_data, title, time_unit="нс", ylabel="Интенсивность"):
    try:
        initial_guess = [max(y_data), len(y_data) / 2, len(y_data) / 4]
        params, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
        A_fit, mu_fit, sigma_fit = params

        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = gaussian(x_fit, A_fit, mu_fit, sigma_fit)

        ax.plot(x_data, y_data, label='Исходные данные')
        ax.plot(x_fit, y_fit, label='Аппроксимация', linestyle='--', color='red')

        param_text = f"A = {A_fit:.2f}\nμ = {mu_fit:.2f}\nσ = {sigma_fit:.2f}"
        ax.text(0.95, 0.95, param_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.7))

        ax.legend()
    except RuntimeError:
        ax.plot(x_data, y_data, label='Исходные данные')
        ax.text(0.5, 0.5, "Аппроксимация не удалась",
                transform=ax.transAxes,
                fontsize=12, color='red',
                ha='center', va='center')

    ax.set_title(title)
    ax.set_xlabel(f"Время, {time_unit}")
    ax.set_ylabel(ylabel)

#Сохранение графика профиля
def save_profile_plot(x_data, y_data, title, time_unit, output_path):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    plot_profile(ax, x_data, y_data, title, time_unit)
    fig.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return output_path

#Сохранение пикселей пучка и его PNG-представления
def save_beam_and_image(beam_crop, beam_prefix, output_dir):
    result_paths = []

    npy_path = os.path.join(output_dir, f"{beam_prefix}.npy")
    np.save(npy_path, beam_crop)
    result_paths.append(npy_path)

    max_val = beam_crop.max()
    safe_max = max_val if max_val > 0 else 1

    norm_img = (beam_crop.astype(np.float32) / safe_max * 255).clip(0, 255).astype(np.uint8)

    png_filename = f"{beam_prefix}_max_value_{int(max_val)}.png"
    png_path = os.path.join(output_dir, png_filename)

    Image.fromarray(norm_img, mode='L').save(png_path)
    result_paths.append(png_path)

    return result_paths

#Сохранение результатов в zip-архив
def save_results_to_zip(file_paths, output_zip_path, base_dir=None):
    try:
        with ZipFile(output_zip_path, 'w') as zipf:
            for file_path in file_paths:
                arcname = (
                    os.path.relpath(file_path, start=base_dir)
                    if base_dir and os.path.commonpath([file_path, base_dir]) == base_dir
                    else os.path.basename(file_path)
                )
                zipf.write(file_path, arcname)
        return output_zip_path
    except Exception as e:
        st.error(f"Ошибка при создании ZIP-архива: {e}")
        return None

#Создание кнопки сохранения результатов
def offer_zip_download(zip_path, download_name):
    with open(zip_path, "rb") as f:
        zip_data = f.read()

    st.download_button(
        label="Скачать все результаты",
        data=zip_data,
        file_name=download_name,
        mime="application/zip",
        type="primary"
    )

#Обработка пучков
def process_detected_beams(beams, filename, k, time_unit, output_dir):
    base_name = os.path.splitext(filename)[0]

    for i, (x, y, w, h, beam_crop, beam_display) in enumerate(beams, 1):
        beam_prefix = f"{base_name}_beam_{x}_{y}_{w}_{h}"

        beam_files = save_beam_and_image(beam_crop, beam_prefix, output_dir)

        col_sums, row_sums, x_horizontal, x_vertical = calculate_profiles(beam_crop, k)

        hor_path = save_profile_plot(
            x_horizontal, col_sums, "Горизонтальный профиль", time_unit,
            os.path.join(output_dir, f"{beam_prefix}_horizontal.png")
        )
        vert_path = save_profile_plot(
            x_vertical, row_sums, "Вертикальный профиль", time_unit,
            os.path.join(output_dir, f"{beam_prefix}_vertical.png")
        )

        beam_files.extend([hor_path, vert_path])

        yield {
            "beam_display": beam_display,
            "horizontal_path": hor_path,
            "vertical_path": vert_path,
            "beam_prefix": beam_prefix,
            "result_files": beam_files
        }

#Обработка одного файла
def process_single_file(file_content, filename, k=1.0, time_unit="нс"):
    file_extension = os.path.splitext(filename)[1].lower()
    original_img, display_img = load_image(file_content, file_extension)
    if original_img is None:
        return []

    prediction = show_image_and_prediction(display_img, original_img)
    k, time_unit = get_processing_params()

    if prediction != 1:
        st.warning("Пучок не обнаружен")
        return []

    beams = detect_beams(original_img)
    if not beams:
        st.info("Пучки не найдены")
        return []

    with tempfile.TemporaryDirectory() as temp_dir:
        result_files = []
        for beam_info in process_detected_beams(beams, filename, k, time_unit, temp_dir):
            st.image(beam_info["beam_display"], caption=beam_info["beam_prefix"], use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(beam_info["vertical_path"], caption="Вертикальный профиль", use_container_width=True)
            with col2:
                st.image(beam_info["horizontal_path"], caption="Горизонтальный профиль", use_container_width=True)

            result_files.extend(beam_info["result_files"])

        zip_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}_results.zip")
        save_results_to_zip(result_files, zip_path)
        offer_zip_download(zip_path, os.path.basename(zip_path))

    return beams

#Обработка ZIP-архива
def process_zip_archive(zip_file, temp_dir):
    k, time_unit = get_processing_params()

    results_dir = os.path.join(temp_dir, f"{os.path.splitext(zip_file.name)[0]}_results")
    os.makedirs(results_dir, exist_ok=True)

    with ZipFile(zip_file) as zip_ref:
        zip_ref.extractall(temp_dir)

    processed_files = []
    error_count = 0
    all_result_files = []

    for root, _, files in os.walk(temp_dir):
        for file in files:
            if '_beam_' in file or file.startswith('.') or not file.lower().endswith(('.pgm', '.npy')):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                original_img, _ = load_image(file_content, os.path.splitext(file)[1].lower())
                if original_img is None or not preprocess_and_predict(original_img):
                    continue

                beams = detect_beams(original_img)
                if not beams:
                    continue

                for beam_info in process_detected_beams(beams, file, k, time_unit, results_dir):
                    all_result_files.extend(beam_info["result_files"])

                processed_files.append(file)

            except Exception as e:
                st.error(f"Ошибка при обработке файла {file}: {e}")
                error_count += 1

    zip_path = os.path.join(temp_dir, "results.zip")
    zip_result = save_results_to_zip(all_result_files, zip_path, base_dir=results_dir)

    return zip_result, processed_files if zip_result else (None, [])

#Основной интерфейс
def main():
    upload_option = st.radio("Выберите тип загрузки:", ("Один файл", "Папка (ZIP архив)"),
                             horizontal=True)
    if upload_option == "Один файл":
        handle_single_file()
    else:
        handle_zip_archive()

if __name__ == "__main__":
    main()