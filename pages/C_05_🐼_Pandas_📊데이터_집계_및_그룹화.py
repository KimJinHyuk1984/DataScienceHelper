# pages/5_📊_데이터_집계_및_그룹화.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.utils_pandas import display_dataframe_info

st.header("5. 데이터 집계 및 그룹화")
st.markdown("""
Pandas는 데이터를 요약하고 통계량을 계산하는 다양한 집계 기능과, 특정 기준에 따라 데이터를 그룹으로 묶어 분석하는 강력한 `groupby` 기능을 제공합니다.
""")

# --- 예제 DataFrame 생성 ---
@st.cache_data
def create_sample_agg_group_df():
    data = {
        'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT', 'IT', 'Sales', 'IT'],
        'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi'],
        'Salary': [70000, 80000, 60000, 65000, 90000, 95000, 75000, 100000],
        'YearsExperience': [5, 7, 3, 4, 8, 10, 6, 12],
        'ProjectsCompleted': [10, 12, 5, 7, 15, 20, 11, 22]
    }
    return pd.DataFrame(data)

sample_df_agg = create_sample_agg_group_df()

st.subheader("집계/그룹화 예제용 DataFrame 확인")
if st.checkbox("집계/그룹화 예제 DataFrame 보기", key="show_agg_base_df_page"):
    display_dataframe_info(sample_df_agg, "예제 DataFrame (sample_df_agg)", max_rows_to_display=8)

st.markdown("---")

# --- 5.1 기본 기술 통계 (Descriptive Statistics) ---
st.subheader("5.1 기본 기술 통계")
st.markdown("""
DataFrame이나 Series의 전체적인 특징을 요약하는 통계량을 계산합니다.
- `df.describe()`: 숫자형 데이터에 대한 주요 기술 통계량 (개수, 평균, 표준편차, 최소값, 사분위수, 최대값)을 반환합니다. `include='all'`로 모든 타입 컬럼 요약 가능.
- 개별 통계 함수: `mean()`, `sum()`, `min()`, `max()`, `count()`, `median()`, `std()`, `var()`, `nunique()` (고유값 개수), `value_counts()` (고유값별 빈도수, Series에 사용).
""")
code_descriptive_stats = """
import pandas as pd
# sample_df_agg DataFrame이 이미 있다고 가정

# 전체 기술 통계
desc_all = sample_df_agg.describe(include='all')
# display_dataframe_info(desc_all, "df.describe(include='all') 결과", display_content=False) # 내용은 st.dataframe으로 별도 표시

# 'Salary' 열의 평균
avg_salary = sample_df_agg['Salary'].mean()
# print(f"'Salary' 평균: {avg_salary:.2f}")

# 'Department' 열의 고유값별 빈도수
dept_counts = sample_df_agg['Department'].value_counts()
# print("\\n'Department' 빈도수:\\n", dept_counts)

# 'ProjectsCompleted'의 총합
total_projects = sample_df_agg['ProjectsCompleted'].sum()
# print(f"\\n'ProjectsCompleted' 총합: {total_projects}")
"""
st.code(code_descriptive_stats, language='python')

if st.checkbox("기본 기술 통계 예시 보기", key="desc_stats_page"):
    st.write("`sample_df_agg.describe(include='all')`:")
    st.dataframe(sample_df_agg.describe(include='all'))
    st.markdown("---")
    st.write(f"`sample_df_agg['Salary'].mean()` (급여 평균): {sample_df_agg['Salary'].mean():,.0f}")
    st.write(f"`sample_df_agg['YearsExperience'].max()` (최대 경력): {sample_df_agg['YearsExperience'].max()}")
    st.write("`sample_df_agg['Department'].nunique()` (부서 고유값 개수):")
    st.write(sample_df_agg['Department'].nunique())
    st.write("`sample_df_agg['Department'].value_counts()` (부서별 직원 수):")
    st.dataframe(sample_df_agg['Department'].value_counts().rename_axis('Department').reset_index(name='Count'))

st.markdown("---")

# --- 5.2 GroupBy (데이터 그룹화) ---
st.subheader("5.2 `groupby()`: 데이터 그룹화")
st.markdown("""
특정 열(들)의 값을 기준으로 데이터를 그룹으로 나누고, 각 그룹에 대해 집계 함수를 적용합니다. 'Split-Apply-Combine' 패턴을 따릅니다.
1.  **Split:** 특정 기준에 따라 데이터를 여러 그룹으로 분할합니다.
2.  **Apply:** 각 그룹에 대해 함수(집계, 변환, 필터링 등)를 독립적으로 적용합니다.
3.  **Combine:** Apply 결과를 합쳐서 새로운 데이터 구조로 만듭니다.
""")
code_groupby = """
import pandas as pd
# sample_df_agg DataFrame이 이미 있다고 가정

# 'Department' 별로 그룹화
grouped_by_dept = sample_df_agg.groupby('Department')

# 각 부서별 평균 급여 계산
avg_salary_by_dept = grouped_by_dept['Salary'].mean()
# print("부서별 평균 급여:\\n", avg_salary_by_dept)

# 각 부서별 직원 수 (size() 사용)
employee_count_by_dept = grouped_by_dept.size().rename('EmployeeCount')
# print("\\n부서별 직원 수:\\n", employee_count_by_dept)

# 여러 기준으로 그룹화: 'Department'와 'YearsExperience' 구간별
# 예시를 위해 YearsExperience를 구간으로 나눔
bins = [0, 5, 10, np.inf] # 0-5년, 6-10년, 10년 초과
labels = ['Junior', 'Mid-Level', 'Senior']
sample_df_agg['ExperienceLevel'] = pd.cut(sample_df_agg['YearsExperience'], bins=bins, labels=labels, right=True)

grouped_multi = sample_df_agg.groupby(['Department', 'ExperienceLevel'])
avg_projects_by_multi = grouped_multi['ProjectsCompleted'].mean()
# print("\\n부서 및 경력 수준별 평균 프로젝트 수:\\n", avg_projects_by_multi)

# agg()를 사용하여 여러 집계 함수 동시 적용
agg_functions = {
    'Salary': ['mean', 'min', 'max'],
    'ProjectsCompleted': 'sum'
}
dept_agg_summary = grouped_by_dept.agg(agg_functions)
# print("\\n부서별 급여(평균,최소,최대) 및 프로젝트 총합:\\n", dept_agg_summary)
"""
st.code(code_groupby, language='python')

if st.checkbox("`groupby()` 예시 보기", key="groupby_page"):
    st.write("`sample_df_agg.groupby('Department')['Salary'].mean()` (부서별 평균 급여):")
    avg_salary_dept = sample_df_agg.groupby('Department')['Salary'].mean().round(0).astype(int)
    st.dataframe(avg_salary_dept.rename_axis('Department').reset_index(name='Average Salary'))

    st.markdown("---")
    st.write("`sample_df_agg.groupby('Department').agg({'YearsExperience': 'mean', 'ProjectsCompleted': ['sum', 'count']})` (부서별 다중 집계):")
    multi_agg_dept = sample_df_agg.groupby('Department').agg(
        Avg_Experience=('YearsExperience', 'mean'),
        Total_Projects=('ProjectsCompleted', 'sum'),
        Num_Employees=('Employee', 'count') # 'Employee' 컬럼으로 직원 수 count
    ).round({'Avg_Experience': 1}) # 평균 경력만 소수점 1자리
    st.dataframe(multi_agg_dept)

    st.markdown("---")
    st.write("그룹화 후 특정 그룹 선택 (`get_group()`): `grouped.get_group('IT')`")
    grouped_by_dept_ex = sample_df_agg.groupby('Department')
    it_dept_df = grouped_by_dept_ex.get_group('IT')
    display_dataframe_info(it_dept_df, "IT 부서 데이터", max_rows_to_display=len(it_dept_df), show_describe=False)


st.markdown("---")

# --- 5.3 피벗 테이블 (Pivot Tables) ---
st.subheader("5.3 피벗 테이블 (`pd.pivot_table()`)")
st.markdown("""
데이터를 재구성하여 요약 테이블을 만듭니다. `groupby`와 유사하지만, 결과를 보다 읽기 쉬운 형태로 보여줍니다.
- `values`: 집계할 값들이 있는 열.
- `index`: 피벗 테이블의 행 인덱스로 사용할 열.
- `columns`: 피벗 테이블의 열로 사용할 열.
- `aggfunc`: 적용할 집계 함수 (기본값: `mean`). 딕셔너리 형태로 여러 함수 지정 가능.
- `fill_value`: 결과 테이블의 NaN 값을 대체할 값.
- `margins`: 부분합/총합 (All) 표시 여부 (기본값 `False`).
""")
code_pivot_table = """
import pandas as pd
import numpy as np
# sample_df_agg DataFrame 및 'ExperienceLevel' 컬럼이 있다고 가정
# bins = [0, 5, 10, np.inf]; labels = ['Junior', 'Mid-Level', 'Senior']
# sample_df_agg['ExperienceLevel'] = pd.cut(sample_df_agg['YearsExperience'], bins=bins, labels=labels)


# 부서(행)별, 경력 수준(열)별 평균 급여 피벗 테이블
pivot_avg_salary = pd.pivot_table(
    sample_df_agg,
    values='Salary',
    index='Department',
    columns='ExperienceLevel',
    aggfunc='mean',
    fill_value=0 # NaN은 0으로 채움
)
# print("부서별, 경력 수준별 평균 급여:\\n", pivot_avg_salary.round(0))

# 여러 값을 집계하고, 총합(margins) 표시
pivot_multi_agg = pd.pivot_table(
    sample_df_agg,
    values=['Salary', 'ProjectsCompleted'],
    index='Department',
    aggfunc={'Salary': np.mean, 'ProjectsCompleted': [min, max, np.sum]},
    margins=True, # 행과 열에 대한 부분합/총합 (All) 추가
    margins_name='Total_Overall'
)
# print("\\n부서별 급여(평균) 및 프로젝트(최소,최대,총합) - 총계 포함:\\n", pivot_multi_agg.round(0))
"""
st.code(code_pivot_table, language='python')

if st.checkbox("`pd.pivot_table()` 예시 보기", key="pivot_table_page"):
    # 예시를 위해 ExperienceLevel 컬럼 추가 (groupby 예시와 동일)
    bins_ex = [0, 5, 10, np.inf]
    labels_ex = ['Junior', 'Mid-Level', 'Senior']
    sample_df_agg['ExperienceLevel'] = pd.cut(sample_df_agg['YearsExperience'], bins=bins_ex, labels=labels_ex, right=True, include_lowest=True)
    
    st.write("`pd.pivot_table(values='Salary', index='Department', columns='ExperienceLevel', aggfunc='mean')` (부서별, 경력수준별 평균 급여):")
    pivot_salary_ex = pd.pivot_table(
        sample_df_agg, values='Salary', index='Department', columns='ExperienceLevel',
        aggfunc='mean', fill_value=0
    ).round(0)
    st.dataframe(pivot_salary_ex)

    st.markdown("---")
    st.write("`pd.pivot_table(values=['Salary', 'ProjectsCompleted'], index='Department', aggfunc={'Salary': 'median', 'ProjectsCompleted': 'sum'}, margins=True)` (부서별 급여 중앙값, 프로젝트 총합, 전체 요약 포함):")
    pivot_multi_ex = pd.pivot_table(
        sample_df_agg,
        values=['Salary', 'ProjectsCompleted'],
        index='Department',
        aggfunc={'Salary': 'median', 'ProjectsCompleted': 'sum'},
        margins=True,
        margins_name='Grand Total'
    )
    st.dataframe(pivot_multi_ex)


st.markdown("---")

# --- 5.4 교차표 (Crosstabulation) ---
st.subheader("5.4 교차표 (`pd.crosstab()`)")
st.markdown("""
두 개 이상의 요인(범주형 변수)에 대한 빈도수를 계산하여 교차표(분할표)를 만듭니다.
- `index`: 교차표의 행으로 사용할 값.
- `columns`: 교차표의 열로 사용할 값.
- `values` (선택 사항): `aggfunc`와 함께 사용하여 빈도수 대신 다른 값을 집계.
- `aggfunc` (선택 사항): `values`가 지정된 경우 적용할 집계 함수.
- `margins`: 행/열 부분합 표시 여부.
- `normalize`: 비율로 정규화 ('index', 'columns', 'all', 또는 `True`는 전체 비율).
""")
code_crosstab = """
import pandas as pd
# sample_df_agg DataFrame 및 'ExperienceLevel' 컬럼이 있다고 가정

# 부서별, 경력 수준별 직원 수 교차표
crosstab_counts = pd.crosstab(
    index=sample_df_agg['Department'],
    columns=sample_df_agg['ExperienceLevel'],
    margins=True, # 행/열 부분합(All) 추가
    margins_name="Total_Count"
)
# print("부서별, 경력 수준별 직원 수 교차표:\\n", crosstab_counts)

# 부서별, 경력 수준별 평균 급여 교차표
crosstab_avg_salary = pd.crosstab(
    index=sample_df_agg['Department'],
    columns=sample_df_agg['ExperienceLevel'],
    values=sample_df_agg['Salary'], # 집계할 값
    aggfunc='mean'                 # 집계 함수
).round(0)
# print("\\n부서별, 경력 수준별 평균 급여 교차표:\\n", crosstab_avg_salary)

# 비율로 정규화 (행 기준)
crosstab_normalized = pd.crosstab(
    index=sample_df_agg['Department'],
    columns=sample_df_agg['ExperienceLevel'],
    normalize='index' # 각 행의 합이 1이 되도록 정규화
).round(2)
# print("\\n부서별, 경력 수준별 직원 비율 (행 기준):\\n", crosstab_normalized)
"""
st.code(code_crosstab, language='python')

if st.checkbox("`pd.crosstab()` 예시 보기", key="crosstab_page"):
    # ExperienceLevel 컬럼이 있는지 확인하고 없으면 생성 (pivot_table 예시와 동일)
    if 'ExperienceLevel' not in sample_df_agg.columns:
        bins_ex = [0, 5, 10, np.inf]
        labels_ex = ['Junior', 'Mid-Level', 'Senior']
        sample_df_agg['ExperienceLevel'] = pd.cut(sample_df_agg['YearsExperience'], bins=bins_ex, labels=labels_ex, right=True, include_lowest=True)

    st.write("`pd.crosstab(index=sample_df_agg['Department'], columns=sample_df_agg['ExperienceLevel'], margins=True)` (부서별, 경력수준별 직원 수):")
    crosstab_counts_ex = pd.crosstab(
        index=sample_df_agg['Department'],
        columns=sample_df_agg['ExperienceLevel'],
        margins=True,
        margins_name="Total"
    )
    st.dataframe(crosstab_counts_ex)

    st.markdown("---")
    st.write("`pd.crosstab(index=sample_df_agg['Department'], columns=sample_df_agg['ExperienceLevel'], values=sample_df_agg['Salary'], aggfunc='mean').round(0)` (부서별, 경력수준별 평균 급여):")
    crosstab_salary_ex = pd.crosstab(
        index=sample_df_agg['Department'],
        columns=sample_df_agg['ExperienceLevel'],
        values=sample_df_agg['Salary'],
        aggfunc='mean'
    ).round(0)
    st.dataframe(crosstab_salary_ex)

    st.markdown("---")
    st.write("`pd.crosstab(index=sample_df_agg['Department'], columns=sample_df_agg['ExperienceLevel'], normalize='columns').round(2)` (열 기준 비율):")
    crosstab_norm_ex = pd.crosstab(
        index=sample_df_agg['Department'],
        columns=sample_df_agg['ExperienceLevel'],
        normalize='columns' # 각 열의 합이 1이 되도록 비율 계산
    ).round(2)
    st.dataframe(crosstab_norm_ex)