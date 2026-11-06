import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_probabilities(_model, features):
    return _model.predict_proba(features)[:, 1]


st.set_page_config(page_title="Подбор вин", layout="wide")

model = load_model()
test_data = load_test_data()
default_threshold = float(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["threshold"])
feature_columns = test_data.drop(columns="target").columns.tolist()
test_features = test_data[feature_columns]
test_targets = test_data["target"].to_numpy()
test_scores = test_probabilities(model, test_features)

threshold = st.sidebar.slider(
    "Порог рекомендации",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.000001,
    format="%.6f",
    key="threshold",
)

st.title("Подбор качественных вин")
st.write("Модель оценивает вероятность того, что вино получит оценку не ниже 7.")

single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs(
    ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

with single_tab:
    st.subheader("Оценка одного вина")
    left, right = st.columns(2)
    input_values = {}
    numeric_features = [name for name in feature_columns if name != "is_red"]

    for index, name in enumerate(numeric_features):
        values = test_data[name]
        column = left if index % 2 == 0 else right
        with column:
            input_values[name] = st.number_input(
                name,
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=float(values.median()),
                step=0.0001 if name == "density" else 0.01,
                format="%.4f",
            )

    input_values["is_red"] = int(
        st.toggle("Красное вино", value=bool(test_data["is_red"].mode().iat[0]))
    )
    wine = pd.DataFrame([input_values], columns=feature_columns)
    probability = float(model.predict_proba(wine)[0, 1])
    st.metric("Вероятность хорошего вина", f"{probability:.1%}")

    if probability >= threshold:
        st.success("Модель рекомендует это вино")
    else:
        st.info("Модель не рекомендует это вино")

with metrics_tab:
    st.subheader("Модельные метрики на тестовой выборке")
    test_predictions = (test_scores >= threshold).astype(int)
    confusion = metrics.confusion_matrix(test_targets, test_predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    precision = metrics.precision_score(test_targets, test_predictions, zero_division=0)
    recall = metrics.recall_score(test_targets, test_predictions)
    f1 = metrics.f1_score(test_targets, test_predictions, zero_division=0)

    precision_column, recall_column, f1_column = st.columns(3)
    precision_column.metric("Precision", f"{precision:.3f}")
    recall_column.metric("Recall", f"{recall:.3f}")
    f1_column.metric("F1", f"{f1:.3f}")

    matrix_figure, matrix_axis = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Остальные", "Хорошие"],
    ).plot(ax=matrix_axis, cmap="Blues", colorbar=False)
    matrix_axis.set_title("Матрица ошибок")
    st.pyplot(matrix_figure, width=560)
    plt.close(matrix_figure)

    fpr, tpr, _ = metrics.roc_curve(test_targets, test_scores)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(test_targets, test_scores)
    current_fpr = fp / (fp + tn)

    curves_figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 4))
    roc_axis.plot(fpr, tpr, label="Модель")
    roc_axis.plot([0, 1], [0, 1], linestyle=":", color="gray")
    roc_axis.scatter(current_fpr, recall, s=80, color="darkorange", label="Текущий порог")
    roc_axis.set(xlabel="False positive rate", ylabel="Recall", title="ROC-кривая")
    roc_axis.legend()

    pr_axis.plot(pr_recall, pr_precision, label="Модель")
    pr_axis.axhline(test_targets.mean(), linestyle=":", color="gray", label="Случайная модель")
    pr_axis.scatter(recall, precision, s=80, color="darkorange", label="Текущий порог")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="PR-кривая")
    pr_axis.legend()

    curves_figure.tight_layout()
    st.pyplot(curves_figure, width=1050)
    plt.close(curves_figure)

    st.write(
        "Если повысить порог, в подборку попадёт меньше вин: снизится число ошибочных "
        "рекомендаций, но часть хороших вин будет пропущена. Если снизить порог, "
        "подборка станет шире: хороших вин найдётся больше, но ошибочных рекомендаций "
        "тоже может стать больше."
    )

with batch_tab:
    st.subheader("Пакетное предсказание")
    st.write("Загрузите CSV с признаками вин. Решения рассчитываются с порогом из боковой панели.")
    uploaded_file = st.file_uploader("CSV-файл", type="csv")

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Не удалось прочитать CSV: {error}")
        else:
            missing_columns = [name for name in feature_columns if name not in uploaded_data.columns]

            if missing_columns:
                st.error("В файле не хватает столбцов: " + ", ".join(missing_columns))
            elif uploaded_data.empty:
                st.error("В файле нет строк с винами")
            else:
                try:
                    uploaded_features = uploaded_data[feature_columns].apply(pd.to_numeric, errors="raise")
                    if not np.isfinite(uploaded_features.to_numpy(dtype=float)).all():
                        raise ValueError("в признаках есть пустые или бесконечные значения")
                    uploaded_scores = model.predict_proba(uploaded_features)[:, 1]
                except (ValueError, TypeError) as error:
                    st.error(f"Не удалось обработать признаки: {error}")
                else:
                    result = uploaded_data.copy()
                    result["probability_good"] = uploaded_scores
                    result["recommend"] = (uploaded_scores >= threshold).astype(int)
                    st.dataframe(result, width="stretch")
                    st.download_button(
                        "Скачать результат",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

with coefficients_tab:
    st.subheader("Что выучила модель")
    coefficients = pd.Series(
        model.named_steps["logisticregression"].coef_[0],
        index=feature_columns,
    ).sort_values()

    coefficient_figure, coefficient_axis = plt.subplots(figsize=(8, 6))
    colors = np.where(coefficients.to_numpy() >= 0, "#2b8a3e", "#c92a2a")
    coefficient_axis.barh(coefficients.index, coefficients.to_numpy(), color=colors)
    coefficient_axis.axvline(0, color="black", linewidth=0.8)
    coefficient_axis.set_xlabel("Коэффициент после масштабирования")
    coefficient_figure.tight_layout()
    st.pyplot(coefficient_figure, width=760)
    plt.close(coefficient_figure)

    st.write(f"Сильнее всего за хорошее вино говорит признак {coefficients.idxmax()}.")
    st.write(f"Сильнее всего против говорит признак {coefficients.idxmin()}.")
