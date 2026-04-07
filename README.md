# FastAI - F1 Race Predictor 🏎️

Sistema de predicción de resultados de carreras de Fórmula 1 basado en Machine Learning.

**Proyecto para Principios y Tecnologias de Inteligencia Artificial - PTIA**  
**Escuela Colombiana de Ingenieria Julio Garavito**

## Desarrollado por:
- David Alejandro Patacon Henao
- Samuel Antonio Gil Romero

## Descripción

Este proyecto utiliza **XGBoost** (Gradient Boosting) para predecir las posiciones finales de los pilotos en carreras de F1. Los datos se obtienen de la API oficial de F1 a través de la librería **FastF1**.

## Características

- 📊 **Datos reales**: Utiliza la API FastF1 para obtener datos oficiales de F1
- 🤖 **XGBoost Regressor**: Modelo de gradient boosting con regularización
- 📈 **18 features**: Incluyendo clasificación, históricos de piloto, equipo y circuito
- 🔄 **Validación temporal**: Time-series cross-validation para evitar data leakage
- 📉 **Métricas esenciales**: MAE, RMSE, Top-3 Accuracy

## Estructura del Proyecto

```
f1_predictor/
├── config/
│   ├── settings.py          # Configuración centralizada
│   └── circuits.py          # Metadata de circuitos
├── data/
│   ├── data_loader.py       # Carga datos desde FastF1
│   └── cache/               # Cache de FastF1
├── features/
│   └── engineering.py       # Feature engineering (18 features)
├── models/
│   ├── trainer.py           # Entrenamiento XGBoost
│   └── saved/               # Modelos guardados
├── evaluation/
│   └── metrics.py           # Métricas de evaluación
├── outputs/
│   └── reports/             # Reportes JSON
├── main.py                  # Pipeline de entrenamiento
├── predict.py               # Script de predicción
├── requirements.txt
└── README.md
```

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Entrenar el modelo

```bash
# Entrenar con datos de 2023, 2024, 2025

# Especificar temporadas
python main.py --seasons 2023 2024 2025
```

### Predecir una carrera

Cabe aclarar que antes de hacer una prediccion ya debe de haber pasado la ronda de clasificacion real para poder cargar los datos desde FastF1 API, de lo contrario dara error y no generar la prediccion

```bash
# Predecir usando nombre de carrera y año
python predict.py --race "Monaco" --year 2026
```

## Features del Modelo

### Features de Clasificación (4)
| Feature | Descripción |
|---------|-------------|
| `quali_position` | Posición en clasificación |
| `quali_gap_to_pole` | Diferencia en segundos al poleman |
| `quali_gap_to_teammate` | Diferencia vs compañero |
| `made_q3` | Si llegó a Q3 |

### Features Históricas de Piloto (4)
| Feature | Descripción |
|---------|-------------|
| `driver_avg_position_last_5` | Promedio últimas 5 carreras |
| `driver_circuit_avg_position` | Promedio en ese circuito |
| `driver_dnf_rate` | Tasa de abandonos |
| `driver_experience` | Número de carreras |

### Features de Equipo (3)
| Feature | Descripción |
|---------|-------------|
| `team_avg_position_season` | Promedio del equipo |
| `team_reliability_rate` | Fiabilidad del equipo |
| `constructor_standing` | Posición en campeonato |

### Features de Circuito (4)
| Feature | Descripción |
|---------|-------------|
| `circuit_type` | Tipo (calle/permanente/híbrido) |
| `circuit_length_km` | Longitud en km |
| `overtaking_difficulty` | Dificultad adelantamiento (1-5) |
| `number_of_laps` | Número de vueltas |

### Features de Grid y Condiciones (3)
| Feature | Descripción |
|---------|-------------|
| `grid_position` | Posición de salida |
| `is_wet_session` | Lluvia prevista |
| `temperature` | Temperatura ambiente |

## Métricas de Evaluación

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **MAE** | Mean Absolute Error | < 3.5 posiciones |
| **RMSE** | Root Mean Squared Error | < 4.5 posiciones |
| **Top-3 Accuracy** | Acierto en podio | > 60% |

## Estrategia de Validación

```
Temporada 2023        Temporada 2024           Test
┌──────────────┐    ┌──────────────────────┐  ┌────────────┐
│  TRAINING    │    │ TRAINING │ VALIDATION│  │   TEST     │
│  22 carreras │    │ 20 carr. │  4 carr.  │  │ 4 carreras │
└──────────────┘    └──────────────────────┘  └────────────┘
```

- **Time-Series Cross-Validation**: 5 folds con ordenamiento temporal
- **Sin data leakage**: Features históricas calculadas solo con datos pasados
- **Test final**: Últimas 4 carreras de la temporada

## Hiperparámetros del Modelo

```python
XGBOOST_PARAMS = {
    'n_estimators': 150,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}
```

**Justificación:**
- `n_estimators=150`: Balance entre rendimiento y overfitting
- `max_depth=4`: Previene árboles excesivamente complejos
- `learning_rate=0.05`: Aprendizaje conservador para mejor generalización
- `reg_alpha/lambda`: Regularización L1/L2 para suavizar predicciones

## Tecnologías Utilizadas

- **Python 3.9+**
- **XGBoost**: Gradient boosting
- **FastF1**: API oficial de datos de F1
- **Pandas/NumPy**: Manipulación de datos
- **Scikit-learn**: Utilidades de ML
- **Matplotlib/Seaborn**: Visualización


## EJEMPLO DE USO
Entrenar modelo:
python main.py --seasons 2023 2024 2025

Output:
```
Summary:
  - Model: XGBoost Regressor
  - Training samples: 1019
  - Test samples: 79
  - Features: 18
  
Test Set Performance:
  - MAE: 2.90 positions
  - RMSE: 3.82 positions
  - Top-3 Accuracy: 71.4% 
```

Predecir una carrera:

python predict.py --race "Japon" --year 2026

Output
```
╔═══════════════════════════════════════════════════════════════╗
║                    F1 RACE PREDICTION                         ║
╠═══════════════════════════════════════════════════════════════╣
║  Japanese Grand Prix                                          ║
║  Season 2026                                                  ║
╚═══════════════════════════════════════════════════════════════╝
    
=================================================================
 POS DRIVER TEAM                        GRID    SCORE
=================================================================
  P1 ANT    Mercedes                       1     2.60
  P2 PIA    McLaren                        3     3.37
  P3 LEC    Ferrari                        4     3.93
 P04 RUS    Mercedes                       2     3.94
 P05 NOR    McLaren                        5     4.62
 P06 HAM    Ferrari                        6     5.35
 P07 HAD    Red Bull Racing                8     7.01
 P08 VER    Red Bull Racing               11     9.28
 P09 GAS    Alpine                         7    10.19
 P10 BOR    Audi                           9    11.54
 P11 LIN    Racing Bulls                  10    12.42
 P12 COL    Alpine                        15    13.45
 P13 OCO    Haas F1 Team                  12    13.55
 P14 LAW    Racing Bulls                  14    13.78
 P15 HUL    Audi                          13    14.00
 P16 SAI    Williams                      16    15.19
 P17 BEA    Haas F1 Team                  18    15.26
 P18 ALB    Williams                      17    15.97
 P19 PER    Cadillac                      19    16.30
 P20 STR    Aston Martin                  22    16.42
 P21 BOT    Cadillac                      20    16.64
 P22 ALO    Aston Martin                  21    17.52
=================================================================

Resultados reales:  
```
>El modelo predijo correctamente el resultado de las primeras 5 posiciones del grand Prix de japon 2026
