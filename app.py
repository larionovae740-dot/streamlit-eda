import pickle
import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

MODEL_PATH = Path(__file__).resolve().parent / "model_bundle.pkl"
CARS_TRAIN = "https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv"

NUM_IMPUTE = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "max_torque_rpm",
    "seats",
]


def _first_float(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else np.nan


def _parse_torque(val):
    if pd.isna(val):
        return np.nan, np.nan
    s = str(val).strip()
    t = re.search(r"([\d.]+)\s*Nm", s, re.I)
    torque = float(t.group(1)) if t else np.nan
    r = re.search(r"@\s*([\d.]+)\s*-\s*([\d.]+)\s*rpm", s, re.I)
    if r:
        max_rpm = float(r.group(2))
    else:
        r2 = re.search(r"@\s*([\d.]+)\s*rpm", s, re.I)
        max_rpm = float(r2.group(1)) if r2 else np.nan
    return torque, max_rpm


def parse_mileage_engine_power(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mileage"] = out["mileage"].map(_first_float)
    out["engine"] = out["engine"].map(_first_float)
    out["max_power"] = out["max_power"].map(_first_float)
    tqt = out["torque"].map(_parse_torque)
    out["torque"] = [t[0] for t in tqt]
    out["max_torque_rpm"] = [t[1] for t in tqt]
    return out


def _looks_like_raw_kaggle(df: pd.DataFrame) -> bool:
    if "name" in df.columns and "brand" not in df.columns:
        return True
    if "mileage" in df.columns and df["mileage"].dtype == object:
        return True
    return False


@st.cache_data
def train_reference_medians() -> pd.Series:
    df = pd.read_csv(CARS_TRAIN)
    feat_cols = [c for c in df.columns if c != "selling_price"]
    df = df.drop_duplicates(subset=feat_cols, keep="first")
    df = parse_mileage_engine_power(df)
    return df[NUM_IMPUTE].median()


@st.cache_data
def load_train_raw() -> pd.DataFrame:
    return pd.read_csv(CARS_TRAIN)


@st.cache_resource
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        st.error(
            "Нет `model_bundle.pkl`. Выполните в ноутбуке ячейку с `pickle.dump(bundle, ...)`."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _bundle_scaler(bundle: dict):
    return bundle.get("scaler_ohe_std")


def prepare_dataframe(df: pd.DataFrame, *, raw: bool, medians: pd.Series) -> pd.DataFrame:
    out = df.copy()
    if raw or _looks_like_raw_kaggle(out):
        out = parse_mileage_engine_power(out)
        if "name" in out.columns:
            out["brand"] = out["name"].astype(str).str.split().str[0]
            out = out.drop(columns=["name"], errors="ignore")
    for c in NUM_IMPUTE:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in NUM_IMPUTE:
        if c in out.columns:
            out[c] = out[c].fillna(medians.get(c, np.nan))
    if "engine" in out.columns:
        out["engine"] = out["engine"].round().astype("Int64").astype(int)
    if "seats" in out.columns:
        out["seats"] = out["seats"].round().astype("Int64").astype(int)
    return out


def preprocess_input(df: pd.DataFrame, bundle: dict, *, raw: bool, medians: pd.Series) -> np.ndarray:
    needed = list(bundle["all_cat_block_columns"])
    work = prepare_dataframe(df, raw=raw, medians=medians)
    miss = [c for c in needed if c not in work.columns]
    if miss:
        raise ValueError(f"Не хватает колонок: {miss}")
    work = work[needed]
    Xnum = work[bundle["num_cols_ohe_block"]].to_numpy(dtype=float)
    Xcat = bundle["ohe"].transform(work[bundle["cat_ohe_cols"]].astype(str))
    X = np.hstack([Xnum, Xcat])
    return _bundle_scaler(bundle).transform(X)


def feature_names(bundle: dict) -> list[str]:
    coef = bundle["ridge"].coef_
    try:
        cat_names = list(bundle["ohe"].get_feature_names_out(
            bundle["cat_ohe_cols"]))
    except Exception:
        cat_names = [f"ohe_{i}" for i in range(
            len(coef) - len(bundle["num_cols_ohe_block"]))]
    names = list(bundle["num_cols_ohe_block"]) + cat_names
    if len(names) != len(coef):
        names = [f"f{i}" for i in range(len(coef))]
    return names


st.set_page_config(page_title="Прогноз цены авто",
                   layout="wide")
st.title("Прогноз стоимости автомобиля")
st.caption(
    "Модель: Ridge + OHE. Поддержка CSV-загрузки, ручного ввода и анализа весов.")

medians = train_reference_medians()
bundle = load_bundle()

with st.sidebar:
    st.header("Как пользоваться")
    st.markdown(
        """
1. **EDA** - обзор train (`cars_train.csv` с GitHub).
2. **Прогноз** - загрузите CSV (сырой как Kaggle или уже с числовыми `mileage`/`engine`/…) **без** обязательной колонки `selling_price`, либо заполните форму.
3. **Веса** - топ коэффициентов Ridge после `StandardScaler` на блоке числа+OHE.
        """
    )
    st.info(
        "Перед первым запуском сохраните `model_bundle.pkl` из ноутбука "
        "(ячейка с `pickle.dump`)."
    )

tab_eda, tab_pred, tab_coef = st.tabs(["EDA", "Прогноз", "Веса модели"])

with tab_eda:
    st.subheader("Информативные графики по обучающей выборке")
    df_demo = load_train_raw()
    df_p = parse_mileage_engine_power(df_demo.copy())
    if "name" in df_p.columns:
        df_p["brand"] = df_p["name"].astype(str).str.split().str[0]
    num_cols = [
        "year", "km_driven", "mileage", "engine",
        "max_power", "torque", "selling_price",
    ]
    num_cols = [c for c in num_cols if c in df_p.columns]

    col_left, col_right = st.columns(2)

    with col_left:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_p["selling_price"], kde=True, ax=ax, color="steelblue")
        ax.set_title("Распределение цены продажи")
        ax.set_xlabel("selling_price")
        ax.set_ylabel("Число объектов")
        st.pyplot(fig)
        plt.close()
        st.markdown(
            "Распределение сильно скошено вправо — большинство автомобилей стоят до 1–2 млн, "
            "но есть редкие экземпляры ценой до 10 млн, которые тянут среднее вверх "
            "(медиана около 450 тыс. против среднего около 640 тыс.). "
            "Из-за этого MSE будет перегибаться под выбросы, "
            "поэтому таргет лучше логарифмировать перед обучением."
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_p.sample(min(2500, len(df_p)), random_state=42),
            x="year", y="selling_price", alpha=0.25, ax=ax, color="steelblue",
        )
        ax.set_title("Цена vs год выпуска")
        ax.set_xlabel("Год выпуска")
        ax.set_ylabel("selling_price")
        st.pyplot(fig)
        plt.close()
        st.markdown(
            "Чем новее автомобиль, тем он дороже, особенно это заметно после 2015 года: "
            "разброс цен резко вырастает. Зависимость монотонная, но не линейная "
            "(Пирсон около 0.43, Спирмен около 0.71), поэтому стоит добавить year^2 в признаки."
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_p.sample(min(2500, len(df_p)), random_state=1),
            x="km_driven", y="selling_price", alpha=0.2, ax=ax, color="coral",
        )
        ax.set_title("Цена vs пробег")
        ax.set_xlabel("km_driven")
        ax.set_ylabel("selling_price")
        st.pyplot(fig)
        plt.close()
        st.markdown(
            "Чем больше пробег, тем ниже цена, но эффект быстро пропадает: первые 100 тыс. км "
            "уменьшают стоимость сильнее, чем следующие 200 тыс. км. Логарифмирование km_driven выровняет зависимость и приглушает эти аномалии."
        )

    with col_right:
        fig, ax = plt.subplots(figsize=(6, 4))
        lp = np.log1p(df_p["selling_price"].clip(lower=0))
        sns.histplot(lp, kde=True, ax=ax, color="darkseagreen")
        ax.set_title("log1p(цена) - после логарифмирования")
        ax.set_xlabel("log1p(selling_price)")
        ax.set_ylabel("Число объектов")
        st.pyplot(fig)
        plt.close()
        st.markdown(
            "После логарифмирования распределение становится заметно симметричнее — "
            "правый хвост пропадает. Модели, обученные на логарифмированной целевой переменной, "
            "дают меньше ошибок на дорогих автомобилях. В нашем случае таргет не логарифмируется, "
            "так что выбросы в верхнем ценовом сегменте будут влиять на метрику сильнее."
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        order = df_p["fuel"].value_counts().index
        sns.boxplot(data=df_p, x="fuel", y="selling_price", order=order, ax=ax,
                    palette="Set2")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_title("Цена по типу топлива")
        ax.set_xlabel("Тип топлива")
        ax.set_ylabel("selling_price")
        st.pyplot(fig)
        plt.close()
        st.markdown(
            "Дизельные автомобили в среднем дороже бензиновых, а газовые — "
            "самые дешевые. "
            "У бензиновых самый широкий разброс — от эконома до премиум-класса. "
            "Тип топлива хорошо разделяет ценовые сегменты."
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sub = df_p[num_cols].dropna()
        if len(sub) > 2:
            cm = sub.corr(numeric_only=True)
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="vlag",
                        center=0, ax=ax, vmin=-1, vmax=1,
                        annot_kws={"size": 8})
            ax.set_title("Корреляция Пирсона: числовые признаки")
        st.pyplot(fig)
        plt.close()
        st.markdown(
            "`engine`, `max_power` и `torque` описывают мощность двигателя и сильно коррелируют между собой."
            "Из-за этой мультиколлинеарности коэффициенты линейной модели становятся нестабильными, "
            "и Ridge здесь надежнее, чем Lasso. "
        )
with tab_pred:
    st.subheader("Предсказание цены")
    mode = st.radio(
        "Формат входных данных",
        ("Авто: CSV сырой",
         "Таблица уже в числовом виде"),
        horizontal=True,
    )
    raw_flag = mode.startswith("Авто")

    up = st.file_uploader("CSV (одна или несколько строк)", type=["csv"])
    if up is not None:
        raw_bytes = up.getvalue()
        try:
            batch = pd.read_csv(BytesIO(raw_bytes))
            Xs = preprocess_input(batch, bundle, raw=raw_flag, medians=medians)
            pred = bundle["ridge"].predict(Xs)
            out = batch.copy()
            out["predicted_price"] = np.round(pred).astype(int)
            st.success(f"Строк: {len(out)}. Первые строки с прогнозом:")
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button(
                "Скачать CSV с прогнозом",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Ошибка: {e}")

    st.markdown("---")
    st.markdown(
        "**Ручной ввод**")
    col_a, col_b = st.columns(2)
    defaults = {
        "brand": "Maruti",
        "year": 2015,
        "km_driven": 50000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": 20.0,
        "engine": 1248,
        "max_power": 88.0,
        "torque": 200.0,
        "max_torque_rpm": 2000.0,
        "seats": 5,
    }
    vals: dict = {}
    for i, (k, v) in enumerate(defaults.items()):
        with col_a if i % 2 == 0 else col_b:
            if isinstance(v, float):
                vals[k] = st.number_input(k, value=float(
                    v), format="%.2f", key=f"mi_{k}")
            elif isinstance(v, int):
                vals[k] = int(st.number_input(
                    k, value=int(v), step=1, key=f"mi_{k}"))
            else:
                vals[k] = st.text_input(k, value=str(v), key=f"mi_{k}")
    button = st.button("Спрогнозировать", type="primary")
    if button:
        row = pd.DataFrame([vals])
        try:
            Xs = preprocess_input(row, bundle, raw=False, medians=medians)
            p = float(bundle["ridge"].predict(Xs)[0])
            st.success(f"Прогноз цены: **{f'{p:,.0f}'.replace(',', ' ')}**")
        except Exception as e:
            st.error(str(e))

with tab_coef:
    st.subheader("Коэффициенты модели")
    ridge = bundle["ridge"]
    names = feature_names(bundle)
    wdf = pd.DataFrame({"feature": names, "weight": ridge.coef_})
    wdf["abs_w"] = wdf["weight"].abs()
    st.metric("Intercept", f"{ridge.intercept_:,.0f}")
    top_n = st.slider("Сколько топ-признаков на графике", 15, 60, 30, 5)
    top = wdf.sort_values("abs_w", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.22 * top_n)))
    sns.barplot(data=top, y="feature", x="weight", ax=ax, color="steelblue")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(f"Топ-{top_n} по |коэффициенту|")
    st.pyplot(fig)
    plt.close()
    st.dataframe(wdf.sort_values("abs_w", ascending=False).head(
        80), use_container_width=True)
    st.download_button(
        "Скачать все коэффициенты CSV",
        data=wdf.sort_values("abs_w", ascending=False).to_csv(
            index=False).encode("utf-8"),
        file_name="ridge_coefficients.csv",
        mime="text/csv",
    )
