# Laboratorio #6 — Transfer Learning y Fine-Tuning

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Nicolás Concuá**

Comparación de tres estrategias para clasificar **CIFAR-10**:

1. **CNN desde cero** (resolución nativa 32×32, BatchNorm + dropout + data augmentation).
2. **VGG-16 como feature extractor** (pesos de ImageNet, backbone congelado, solo se entrena el clasificador).
3. **VGG-16 con fine-tuning** (bloques convolucionales superiores descongelados, learning rate menor para el backbone).

Ambos modelos VGG-16 usan entrada de **112×112** (4× menos cómputo que 224×224) por restricción de hardware.
Se registran 10 iteraciones (3 + 3 + 4), métricas macro, parámetros, tiempo, memoria, MACs/FLOPs, latencia,
cómputo total de entrenamiento y un experimento con el 10 % de los datos.

## Estructura

```
Lab6-DL/
├── notebook/
│   └── lab6.ipynb      # notebook completo: datos, investigación, modelos, entrenamiento, recursos, discusión
├── reports/            # figuras (.png), tablas (.csv), resultados (.json) e informe
└── data/               # CIFAR-10 (se descarga automáticamente, no se versiona)
```

## Cómo reproducir

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install torch torchvision torchinfo fvcore scikit-learn pandas matplotlib jupyter
cd notebook
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 lab6.ipynb
```

El notebook detecta CUDA, MPS (Apple Silicon) o CPU. Con `LAB6_QUICK=1` corre una prueba rápida
(subconjuntos pequeños y 1 epoch). Hardware usado: Apple M4 Pro (MPS).
