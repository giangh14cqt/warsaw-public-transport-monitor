# Data Processing, Modeling & Delay Attribution Engine

This segment documents the downstream analytical, fusion, econometric, and machine learning components of the Warsaw Transit Telemetry System. Orchestrated via [`run_pipeline.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/run_pipeline.py) and [`src/pipeline/stages.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/pipeline/stages.py), the data processing pipeline bridges raw telemetry with academic-grade causal and predictive insights.

---

## 🏗️ End-to-End Processing Architecture

```mermaid
%%{init: {
  'theme': 'dark',
  'themeVariables': {
    'darkMode': true,
    'background': '#0d1117',
    'mainBkg': '#161b22',
    'primaryColor': '#1f2937',
    'primaryTextColor': '#f0f6fc',
    'primaryBorderColor': '#58a6ff',
    'lineColor': '#8b949e',
    'secondaryColor': '#161b22',
    'tertiaryColor': '#21262d',
    'clusterBkg': '#161b22',
    'clusterBorder': '#30363d',
    'edgeLabelBackground': '#161b22'
  }
}}%%
flowchart TD
    %% Stage 1: Exogenous
    subgraph StageExo ["Stage 1: Exogenous Harvesting (src/exogenous/)"]
        direction TB
        IMGW_Poller["IMGW-PIB Synoptic Weather<br/><b>imgw.py</b><br/><i>Temperature, Rain, Wind, Visibility</i>"]
        OSM_Poller["OSMnx Road Corridor Topology<br/><b>osm.py</b><br/><i>Lanes, Max Speed, Corridor Congestion</i>"]
        Hub_Poller["Transit Hub Topology<br/><b>interchange.py</b><br/><i>Metro, Tram & Rail Multi-Modal Hubs</i>"]
        Cal_Poller["Calendar & Holiday Engine<br/><b>src/core/calendar.py</b><br/><i>Polish Public Holidays & Peak Windows</i>"]
        
        ExoWeather[("data/exogenous/weather.parquet")]
        ExoOSM[("data/exogenous/osm_corridors.parquet")]
        ExoHub[("data/exogenous/interchanges.parquet")]

        IMGW_Poller --> ExoWeather
        OSM_Poller --> ExoOSM
        Hub_Poller --> ExoHub
    end

    %% Stage 2: Feature Mart Fusion
    subgraph StageFusion ["Stage 2: DuckDB Vectorized Mart Fusion (src/fusion/)"]
        direction TB
        RawSource[("Telemetry Stream<br/>• data/raw/ (Legacy)<br/>• data/historian/gold/ (Historian)")]
        DuckDBEngine["DuckDB SQL Fusion Engine<br/><b>feature_mart.py</b><br/><i>Lagged Window Joins, ASOF Timestamp Match</i>"]
        MartParquet[("Fused Feature Mart<br/>• feature_mart.parquet<br/>• feature_mart_observed.parquet")]

        RawSource --> DuckDBEngine
        ExoWeather --> DuckDBEngine
        ExoOSM --> DuckDBEngine
        ExoHub --> DuckDBEngine
        Cal_Poller --> DuckDBEngine
        DuckDBEngine --> MartParquet
    end

    %% Stage 3: Purged Validation & Split
    subgraph StageVal ["Stage 3: Purged Temporal Block Validation (src/models/validation.py)"]
        direction TB
        PurgedCV["Purged Forward-Chaining Splitter<br/><i>Train / Validation / Test Blocks</i>"]
        Embargo["Adaptive Embargo Buffer<br/><i>Strict Zero-Leakage Guarantee</i>"]
        Cohorts["Cohort Partitioner<br/><i>Urban Bus | Urban Tram | Suburban Bus | Pooled</i>"]

        MartParquet --> PurgedCV
        PurgedCV --> Embargo --> Cohorts
    end

    %% Stage 4: Econometric & GBDT Modeling
    subgraph StageModels ["Stage 4: Benchmark Modeling Suite (src/models/)"]
        direction TB
        TWFE["Econometric TWFE Baseline<br/><b>twfe.py</b><br/><i>Route & Time Two-Way Fixed Effects</i>"]
        LGBM["LightGBM Regressor<br/><b>gbm.py</b><br/><i>Gradient Boosted Decision Trees</i>"]
        CatB["CatBoost Regressor<br/><b>gbm.py</b><br/><i>Categorical Feature Decision Trees</i>"]
        Metrics["Evaluation Engine (src/models/metrics.py)<br/><i>MAE, RMSE, MedAE, WAPE, R²</i>"]

        Cohorts --> TWFE
        Cohorts --> LGBM
        Cohorts --> CatB
        TWFE --> Metrics
        LGBM --> Metrics
        CatB --> Metrics
    end

    %% Stage 5: xAI Diagnostics
    subgraph StageXAI ["Stage 5: Explainable AI & Policy Diagnostics (src/xai/)"]
        direction TB
        SHAP["TreeSHAP Engine<br/><b>shap_explainer.py</b>"]
        ALE["Accumulated Local Effects (ALE)<br/><b>ale_curves.py</b>"]
        Interactions["Friedman's H-Statistic<br/><b>interactions.py</b>"]
        DiCE["DiCE Counterfactuals<br/><b>counterfactuals.py</b>"]
        PubFigures[("reports/figures/<br/>Publication Charts (300 DPI)")]

        LGBM --> SHAP & ALE & Interactions & DiCE
        CatB --> SHAP & ALE & Interactions & DiCE
        SHAP --> PubFigures
        ALE --> PubFigures
        Interactions --> PubFigures
        DiCE --> PubFigures
    end

    %% Subgraph Styling for Dark Theme
    style StageExo fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style StageFusion fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style StageVal fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style StageModels fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style StageXAI fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    %% High-Contrast Node Styling
    classDef exoStyle fill:#21262d,stroke:#8b949e,stroke-width:2px,color:#f0f6fc;
    classDef fusionStyle fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef valStyle fill:#271d3b,stroke:#bc8cff,stroke-width:2px,color:#f0f6fc;
    classDef modelStyle fill:#33172b,stroke:#f472b6,stroke-width:2px,color:#f0f6fc;
    classDef xaiStyle fill:#33171a,stroke:#ff7b72,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class IMGW_Poller,OSM_Poller,Hub_Poller,Cal_Poller,ExoWeather,ExoOSM,ExoHub exoStyle;
    class RawSource,DuckDBEngine,MartParquet fusionStyle;
    class PurgedCV,Embargo,Cohorts valStyle;
    class TWFE,LGBM,CatB,Metrics modelStyle;
    class SHAP,ALE,Interactions,DiCE,PubFigures xaiStyle;
```

---

## 💻 CLI Orchestrator Dispatcher (`run_pipeline.py`)

All stages are executed synchronously or independently via a unified CLI dispatcher:

```mermaid
%%{init: {
  'theme': 'dark',
  'themeVariables': {
    'darkMode': true,
    'background': '#0d1117',
    'mainBkg': '#161b22',
    'primaryColor': '#16243b',
    'primaryTextColor': '#f0f6fc',
    'primaryBorderColor': '#58a6ff',
    'lineColor': '#8b949e'
  }
}}%%
graph LR
    CLI["run_pipeline.py"] --> S_Pre["--stage preflight<br/><i>Sanity & Environment Audit</i>"]
    CLI --> S_Exo["--stage weather / osm<br/><i>Exogenous Telemetry Harvest</i>"]
    CLI --> S_Fus["--stage fusion<br/><i>Dual-Datasource Mart Fusion</i>"]
    CLI --> S_Bmk["--stage benchmark<br/><i>TWFE vs. LightGBM vs. CatBoost</i>"]
    CLI --> S_Xai["--stage xai<br/><i>Attributions, ALE, DiCE & Plots</i>"]

    classDef cliNode fill:#271d3b,stroke:#bc8cff,stroke-width:2px,color:#f0f6fc;
    classDef stageNode fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class CLI cliNode;
    class S_Pre,S_Exo,S_Fus,S_Bmk,S_Xai stageNode;
```

### CLI Command Reference
```bash
# 1. Harvest hourly meteorological telemetry from IMGW-PIB
python run_pipeline.py --stage weather --start-date 2026-10-01 --end-date 2026-10-18

# 2. Extract OSM road corridor topology
python run_pipeline.py --stage osm

# 3. Fuse feature mart (Legacy GTFS-RT Telemetry)
python run_pipeline.py --stage fusion --datasource legacy

# 4. Fuse feature mart (Historian Observed Actuals)
python run_pipeline.py --stage fusion --datasource historian

# 5. Train & evaluate benchmark models across all cohorts
python run_pipeline.py --stage benchmark --datasource historian --cohort all

# 6. Generate academic xAI figures & counterfactual recommendations
python run_pipeline.py --stage xai --datasource historian --model lgbm
```

---

## 📑 Detailed Architecture Documents in this Segment

1. [`exogenous_enrichment.md`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/data_processing/exogenous_enrichment.md) - Weather, road topology, and transit interchange extraction mechanics.
2. [`feature_mart_fusion.md`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/data_processing/feature_mart_fusion.md) - DuckDB vectorized dual-datasource fusion engine, database matching (Gold stop events to modeling matrix), lagged feature engineering, and design matrix taxonomy.
3. [`modeling_and_xai.md`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/data_processing/modeling_and_xai.md) - Purged temporal block validation, econometric TWFE, GBDTs, and xAI diagnostic suite.
