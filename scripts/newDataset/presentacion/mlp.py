"""Contenido de la presentación de resultados del MLP (Evaluación Parcial 3).

Las cifras principales salen del resumen de la corrida (`ctx.j`) y del resumen de la CNN (`ctx.otra`); las de los
experimentos, que no están en el resumen, son las de las salidas guardadas del notebook de la misma corrida.
"""

from generar_presentacion import RAIZ, cifras, columna, duracion, entero, estado, fecha, figura, leyenda, lista, num, \
    pct, puntos_de, tabla

TITULO = "¿Quién sale en la imagen? — MLP sobre 64 personajes de Los Simpson (EP3)"
DESCRIPCION = ("Resultados del Perceptrón Multicapa optimizado y auditado: representación de la entrada, "
               "regularización, Grid Search, auditoría de sesgos, SHAP y comparación con la CNN.")
BARRA = "MLP sobre 64 personajes de Los Simpson · EP3"
PIE = "MLP · Los Simpson · Evaluación Parcial 3"
SUBTITULO_PORTADA = ("Un Perceptrón Multicapa sobre descriptores de color, optimizado y auditado para reconocer a 64 "
                     "personajes, comparado con una red convolucional entrenada con las mismas imágenes. Corrida del")


def fecha_portada(ctx):
    """Fecha de la corrida que muestra la portada."""
    return fecha(ctx.j["inicio_corrida"])


def final(ctx):
    """Nota al pie de la página, fuera de las diapositivas."""
    return ("Cifras de <span class=\"dato\">PMC/resultados/newDataset/GPU/rendimiento_gpu_128x128.json</span> y de las "
            "salidas de <span class=\"dato\">PMC/notebooks/LosSimpsons_GPU_4060M.ipynb</span>, corrida del "
            f"{fecha(ctx.j['inicio_corrida'])} ({duracion(ctx.j['duracion_corrida'])}). Las figuras las guarda el "
            "notebook en <span class=\"dato\">PMC/docs/newDataset/figuras/</span>. Generada con "
            "<span class=\"dato\">scripts/newDataset/generar_presentacion.py --modelo mlp</span>.")


def _cnn(ctx, clave, formato=num):
    """Cifra de la CNN o «no medida» si su resumen no la tiene."""
    if not ctx.otra or ctx.otra.get(clave) is None:
        return "no medida"
    return formato(ctx.otra[clave])


def diapositivas(ctx):
    """Lista de (sección, título, cuerpo, disposición) de la presentación del MLP."""
    j = ctx.j
    c = ctx.otra or {}
    d = []

    # 2. Problema
    d.append(("Problema", "Catalogar un archivo que crece más rápido que el equipo", columna(
        cifras([
            ("Personajes", entero(j["personajes"]), "una carpeta por personaje"),
            ("Imágenes originales", "56.268", "60,6 % fotogramas de episodio"),
            ("Partición", "70 / 20 / 10", "estratificada, semilla 42"),
            ("Entrada", "370", "histogramas de color de la imagen completa"),
        ]),
        lista([
            "Clasificación multiclase de etiqueta única: el modelo asigna <strong>un personaje</strong> a cada imagen, "
            "con su confianza.",
            "Uso: el modelo propone y una persona valida; el trabajo pasa de <em>etiquetar</em> a <em>revisar</em>.",
            "Un MLP no ve qué píxel está junto a cuál: recibe <strong>descriptores de color y textura</strong> "
            "calculados de la imagen, y la experimentación los compara con los píxeles crudos.",
            "La EP3 optimiza y audita el pipeline de la EP2: el mismo para el MLP y la CNN, con el mismo presupuesto "
            f"para todas las configuraciones ({j['epocas_maximas']} épocas, paciencia {j['paciencia']}).",
        ]),
    ), ""))

    # 3. Datos
    d.append(("Datos y análisis exploratorio", "Dos orígenes y un desbalance de 20 a 1", (
        ctx.fig("eda_origen.png", "Distribución de las imágenes por personaje y por origen", "lamina alta")
        + columna(lista([
            "<strong>34.106 fotogramas de episodio</strong> (60,6 %) y 22.162 imágenes de Kaggle, en la misma "
            "proporción en entrenamiento, validación y test.",
            "De 3.118 imágenes de Homero a 151 de Maggie: <strong>20,6 a 1</strong>. Se compensa con pesos por clase y "
            "se elige por F1 macro.",
            "9.040 imágenes casi idénticas a otra entran con una versión filtrada; 14.077 alteradas van al conjunto de "
            "su original.",
            "16.747 imágenes (23,8 %) tienen a otro personaje en la escena: Lou y Wiggum, Patty y Selma, Chalmers y "
            "Skinner.",
        ], chicos=True))
    ), "dos mitad"))

    # 4. Pipeline
    d.append(("Pipeline integrado", "Nueve etapas, un módulo común a las dos redes", columna(
        ctx.fig("pipeline.png", "Diagrama del pipeline integrado"),
        leyenda("Las funciones viven en <span class=\"dato\">scripts/newDataset/pipeline_comun/</span>, con sus "
                "pruebas, y las importan los notebooks del MLP y de la CNN: lo propio del MLP son los descriptores y "
                "la función que construye el modelo. La partición va antes que los píxeles, y cada configuración "
                "guarda su historial y se reanuda si la corrida se corta."),
    ), ""))

    # 5. Modelo base y descriptores
    d.append(("Modelo base", "Descriptores de la imagen y dos capas densas", (
        tabla(["Bloque", "Qué mide", "Valores"], [
            ["Color HSV", "Tono, saturación y brillo, en la imagen y sus 4 cuadrantes", "170"],
            ["Color Lab", "Color perceptual, en las mismas 5 zonas", "200"],
            ["Textura LBP", "Microtexturas: zonas lisas, rayas, bordes finos", "232"],
            ["Textura Gabor", "Patrones con una orientación y una escala", "96"],
            ["Bordes HOG", "Hacia dónde apuntan los bordes en 8 × 8 celdas", "1.764"],
        ], numericas=(2,), densa=True)
        + columna(cifras([
            ("Parámetros", "1.411.904", "89,4 % en la primera capa"),
            ("Entrada", "2.462", "20 veces menos que los 49.152 píxeles"),
        ], 2), lista([
            "Red base: 2.462 → 512 → 256 → 64, con BatchNorm, ReLU y dropout 0,3; Adam 10⁻³, batch 128, suavizado "
            "0,1 y pesos por clase (raíz).",
            "Los descriptores se estandarizan dentro del modelo, con la media y la desviación de entrenamiento.",
        ], chicos=True))
    ), "dos"))

    # 6. Entrenamiento
    d.append(("Entrenamiento y parada temprana", "La base memoriza: brecha de 0,397", (
        ctx.fig("curvas_modelo_base.png", "Curvas de pérdida y accuracy del modelo base")
        + columna(cifras([
            ("Mejor época", "30", "val_loss 2,151; paró en la 42"),
            ("Brecha train − val", num(j["brecha_train_val_base"]), "0,870 frente a 0,473"),
            ("Épocas ahorradas", "18", "unos 45 segundos"),
            ("Sin parada temprana", "0,432", "brecha en la última época"),
        ], 2), lista(["Con 1,4 millones de parámetros para 49.148 imágenes, la red memoriza: hay que regularizar y "
                      "achicar la entrada."], chicos=True))
    ), "dos figura-ancha"))

    # 7. Regularización
    d.append(("Regularización", "El dropout es imprescindible; L2 no ayuda", (
        tabla(["Técnica", "Experimento", "Δ F1 val.", "Brecha"], [
            ["Dropout", "G1 · sin dropout", "−7,5", "0,216"],
            ["Dropout", "E6 · dropout 0,5", "+0,2", "0,284"],
            ["Batch Normalization", "G2 · sin BatchNorm", "−1,1", "0,367"],
            ["L2", "G3 · L2 10⁻⁴", "−0,7", "0,311"],
            ["Aumento fuera de línea", "A1 · sin alteradas", "−2,6", "0,333"],
            ["Suavizado", "S1 · sin suavizado", "−3,5", "0,147"],
        ], numericas=(2, 3), densa=True)
        + columna(lista([
            "Base: F1 macro de validación 0,458, brecha 0,397, <strong>mismo presupuesto para todas</strong>.",
            "Sin dropout, la validación toca su mínimo en la época 3: la red memoriza al instante (−7,5 puntos).",
            "El suavizado y las alteradas suman 3,5 y 2,6 puntos: cambian lo que la red ve, no restringen los pesos.",
            "L2 resta y más dropout no cambia nada. El aumento en línea de la CNN no aplica: los descriptores se "
            "calculan en la CPU.",
        ], chicos=True))
    ), "dos"))

    # 8. Un factor a la vez
    d.append(("Optimización · un factor a la vez", "La representación de la entrada decide", (
        ctx.fig("experimentos.png", "F1 macro y accuracy de validación de cada configuración")
        + columna(lista([
            "<strong>Solo color</strong> (R2, 370 valores) supera a color y textura por <strong>6,9 puntos</strong> "
            "de F1 con la cuarta parte de los parámetros.",
            "Solo textura (R3) −17,4 y <strong>píxeles crudos</strong> (R1, 25 millones de parámetros) −24,6: la "
            "peor entrada.",
            "Una tercera capa (E2) +1,2; tanh (E3) −7,9; tasa 10⁻⁴ (E4) −3,2. C1 combina solo color, tres capas, "
            "dropout 0,5 y pesos balanceados: +4,7.",
        ], chicos=True))
    ), "dos figura-ancha"))

    # 9. Grid Search
    d.append(("Optimización · Grid Search", "36 combinaciones en 76 minutos", (
        ctx.fig("busqueda.png", "Curvas de las tres mejores configuraciones del Grid Search frente a la base",
                "lamina alta")
        + columna(lista([
            "Tasa {10⁻³, 5·10⁻⁴} × dropout {0,2, 0,3, 0,5} × L2 {0, 10⁻⁵, 10⁻⁴} × batch {128, 512}, sobre la "
            "entrada y la arquitectura de C1.",
            "<strong>El dropout decide:</strong> 0,556 de F1 en promedio con 0,2, 0,546 con 0,3 y 0,493 con 0,5. "
            "Tasa, batch y L2 mueven menos de un punto.",
            "Corrige a un factor a la vez: el dropout 0,5 ayudaba con 2.462 descriptores y sobra con 370.",
            "GS13 repite C1 y da su mismo F1 (0,5052): es reproducible.",
        ], chicos=True))
    ), "dos figura-ancha"))

    # 10. Selección
    d.append(("Selección del modelo", "Gana GS01: solo color, tres capas y dropout 0,2", (
        tabla(["Configuración", "F1 macro val.", "Accuracy val.", "Brecha", "Parada"], [
            ["<strong>GS01 · Grid Search</strong>", "<strong>0,563</strong>", "0,560", "0,337", "tope"],
            ["GS03 · con L2 10⁻⁵", "0,562", "0,561", "0,337", "tope"],
            ["GS21 · tasa 5·10⁻⁴", "0,562", "0,555", "0,330", "tope"],
            ["R2 · solo color", "0,527", "0,521", "0,155", "tope"],
            ["C1 · combinación", "0,505", "0,498", "0,039", "tope"],
            ["Base", "0,458", "0,473", "0,397", "paciencia"],
        ], numericas=(1, 2, 3), densa=True)
        + columna(lista([
            "Se elige por <strong>F1 macro de validación</strong>: promedia los 64 personajes sin ponderar.",
            "Las tres primeras quedan a menos de 0,2 puntos; GS01 es la más simple (sin L2).",
            "+10,5 puntos sobre la base y +5,8 sobre C1, con 1.059.648 parámetros.",
            "Reserva: 35 de las 36 configuraciones llegaron al tope de 60 épocas; GS01, con su mejor época en la "
            "última.",
        ], chicos=True))
    ), "dos"))

    # 11. Test y KPIs
    kpis = [
        ("Cobertura de asistencia", "Top-3", "≥ 0,70", num(j["top3_test"]), j["top3_test"] >= 0.70),
        ("Equidad entre personajes", "F1 macro", "≥ 0,45", num(j["f1_macro_test"]), j["f1_macro_test"] >= 0.45),
        ("Automatización sin revisión", "Accuracy", "≥ 0,50", num(j["accuracy_test"]), j["accuracy_test"] >= 0.50),
        ("Ventaja sobre lo trivial", "Accuracy / mayoritaria", "≥ 4×", "9,4×", True),
        ("Costo de catalogación", "Imágenes por segundo", "≥ 1.000", "1.360", True),
    ]
    d.append(("Evaluación en test", "Los cinco KPIs se cumplen, algunos por poco", columna(
        cifras([
            ("Accuracy", num(j["accuracy_test"]), "35,6 veces el azar"),
            ("F1 macro", num(j["f1_macro_test"]), f"precision {num(j['precision_macro_test'])} · recall "
                                                  f"{num(j['recall_macro_test'])}"),
            ("Top-3", num(j["top3_test"]), "la respuesta entre tres sugerencias"),
            ("ROC-AUC macro", num(j["roc_auc_macro_test"]), "11 personajes bajo 0,90"),
        ]),
        tabla(["KPI", "Métrica", "Umbral", "MLP", ""], [[a, b, cc, v, estado(ok)] for a, b, cc, v, ok in kpis],
              numericas=(2, 3), densa=True),
    ), ""))

    # 12. ROC y personajes
    d.append(("Evaluación por personaje", "Reconoce paletas: falla con quien comparte colores", (
        ctx.fig("curvas_roc.png", "Curvas ROC de los personajes mejor y peor separados")
        + columna(lista([
            "27 de 64 personajes con F1 ≥ 0,60. Los mejores, de paleta inconfundible: Kang y Kodos (0,81), el "
            "capitán McCallister (0,79), Apu (0,71).",
            "Bart, Lisa y Homero, con más de 1.500 imágenes cada uno, quedan en 0,37, 0,41 y 0,43: comparten la piel "
            "amarilla.",
            "Bajo 0,30: Maggie, Todd y Rod Flanders. La correlación entre imágenes de entrenamiento y F1 es de −0,04.",
            "Judge Snyder tiene ROC-AUC de 0,996 y F1 de 0,61: casi siempre es candidato, pero en segundo lugar.",
        ], chicos=True))
    ), "dos mitad"))

    # 13. Confusiones y errores
    d.append(("Análisis de errores", "Paleta compartida, gemelas y escenas", (
        tabla(["Real", "Predicho", "% de la clase"], [
            ["Todd Flanders", "Rod Flanders", "42,5"],
            ["Selma Bouvier", "Patty Bouvier", "33,6"],
            ["Patty Bouvier", "Selma Bouvier", "26,8"],
            ["Lou", "Chief Wiggum", "23,9"],
            ["Louie", "Fat Tony", "21,1"],
            ["Bart Simpson", "Lisa Simpson", "16,8"],
        ], numericas=(2,), densa=True)
        + columna(cifras([
            ("Errores", entero(3141), "44,4 % del test"),
            ("Nombran a otro presente", "653", "20,8 % de los errores"),
            ("Confiados (> 0,80)", entero(j["errores_confiados_test"]), f"{_cnn(ctx, 'errores_confiados_test', entero)} en la CNN"),
        ], 3), lista([
            "Bart y Lisa se confunden entre sí 70 veces: los histogramas no ven la forma del pelo.",
            "La confianza separa aciertos (0,74) de errores (0,42): sirve para mandar lo dudoso a revisión.",
            f"Accuracy por escena, de referencia: {num(j['accuracy_por_escena_test'])}.",
        ], chicos=True))
    ), "dos"))

    # 14. Explicabilidad: permutación y SHAP
    d.append(("Explicabilidad · SHAP", "El tono y la zona en que aparece cada color", (
        ctx.fig("shap_mlp.png", "SHAP del MLP final por canal de color y zona de la imagen", "lamina alta")
        + columna(cifras([
            ("Barajar HSV", "0,08", "F1 macro de test, desde 0,557"),
            ("Barajar Lab", "0,12", "la red usa los dos espacios"),
        ], 2), lista([
            "El canal H (tono) reúne el 24,5 % del |SHAP| y el eje b* de Lab (azul–amarillo) el 19,3 %.",
            "Los cuatro cuadrantes reúnen el 84,8 %; el histograma de la imagen completa, el 15,2 %.",
            "Exacto sobre lo que la red usa, pero lo que usa son proporciones de color: no hay forma ni cara.",
        ], chicos=True))
    ), "dos mitad"))

    # 15. Recreación
    ruta_recreacion = RAIZ / "PMC" / "resultados" / "newDataset" / "recreaciones" / "comparacion.png"
    d.append(("Explicabilidad · recreación", "Lo que aprendió el MLP de cada personaje", columna(
        figura(ruta_recreacion, "Doce personajes recreados desde el MLP de píxeles"),
        lista([
            "Con DeepInversion se genera una imagen de cada personaje solo desde el MLP de píxeles (R1), sin leer el "
            "dataset.",
            "El MLP reconoce las <strong>64 recreaciones</strong> (0,988 de probabilidad media), pero son manchas de "
            "color sin forma: Bart es casi todo rojo; Maggie, amarillo arriba y celeste abajo.",
            "Las dos CNN reconocen 3 y 2 de las 64, lo mismo que el azar: lo que usa el MLP no es lo que usa la CNN.",
        ], chicos=True),
    ), ""))

    # 16. Calibración y umbral
    d.append(("Auditoría · confiabilidad", "Bien calibrado, pero automatiza poco", (
        ctx.fig("auditoria_2.png", "Diagrama de confiabilidad y distribución de la confianza")
        + columna(cifras([
            ("ECE", num(j["ece_test"]), "confianza 0,598 frente a accuracy 0,556"),
            ("Umbral de revisión", num(j["umbral_revision"]), "elegido en validación para un 90 %"),
            ("Resuelve solo", pct(j["cobertura_umbral_test"]), f"del test ({_cnn(ctx, 'cobertura_umbral_test', pct)} en la CNN)"),
            ("Acierto de lo aceptado", pct(j["accuracy_umbral_test"]), "el resto va a revisión humana"),
        ], 2))
    ), "dos figura-ancha"))

    # 17. Sesgos
    d.append(("Auditoría · sesgos por subgrupo", "La familia Simpson, el sesgo propio del MLP", columna(
        ctx.fig("auditoria_3.png", "Accuracy del MLP por subgrupo"),
        tabla(["Variable", "Grupos", "Accuracy", "Brecha"], [
            ["Familia Simpson", "familia / resto",
             f"{num(j['accuracy_familia_simpson_test'])} / {num(j['accuracy_resto_personajes_test'])}", "11,3 pts"],
            ["Origen", "Kaggle / fotograma de episodio",
             f"{num(j['accuracy_kaggle_test'])} / {num(j['accuracy_episodio_test'])}", "11,9 pts"],
            ["Tamaño de la clase", "pequeñas / medianas / grandes", "0,571 / 0,556 / 0,550", "2,1 pts"],
        ], numericas=(2, 3), densa=True),
        leyenda("La familia comparte la piel amarilla y el MLP solo ve el color: la CNN no tiene esa brecha (0,5 "
                "puntos). La brecha por origen es menor que en la CNN porque el MLP también rinde poco con Kaggle."),
    ), ""))

    # 18. Comparación con la CNN
    filas = [
        ["Configuración final", j["configuracion_final"].split(" - ")[0], c.get("configuracion_final", "—")],
        ["Parámetros", entero(j["parametros_modelo_final"]), _cnn(ctx, "parametros_modelo_final", entero)],
        ["Accuracy", num(j["accuracy_test"]), f"<strong>{_cnn(ctx, 'accuracy_test')}</strong>"],
        ["F1 macro", num(j["f1_macro_test"]), f"<strong>{_cnn(ctx, 'f1_macro_test')}</strong>"],
        ["ROC-AUC macro", num(j["roc_auc_macro_test"]), _cnn(ctx, "roc_auc_macro_test")],
        ["Accuracy en alteradas", num(j["accuracy_alteradas_test"]), _cnn(ctx, "accuracy_alteradas_test")],
        ["ECE", num(j["ece_test"]), _cnn(ctx, "ece_test")],
        ["Segundos por época", num(j["benchmark_segundos_por_epoca"], 2),
         _cnn(ctx, "benchmark_segundos_por_epoca", lambda v: num(v, 1))],
        ["Catalogación (img/s)", "1.360", "3.567"],
    ]
    diferencia = puntos_de(c["f1_macro_test"] - j["f1_macro_test"]) if c.get("f1_macro_test") else "—"
    veces = (num(c["benchmark_segundos_por_epoca"] / j["benchmark_segundos_por_epoca"], 1)
             if c.get("benchmark_segundos_por_epoca") else "—")
    d.append(("Comparación con la CNN", "Mismos datos, mismo proceso: solo cambia el modelo", (
        tabla(["Concepto", "MLP", "CNN"], filas, numericas=(1, 2), densa=True)
        + columna(cifras([("F1 macro", diferencia, "puntos a favor de la CNN"),
                          ("Por época", f"×{veces}", "más cara la CNN")], 2),
                  lista([
                      "La CNN aprende sus filtros sobre los píxeles y ve la forma; el MLP combina 370 histogramas.",
                      "La red del MLP infiere 21 veces más rápido, pero sus descriptores en la CPU lo dejan en 1.360 "
                      "imágenes por segundo de extremo a extremo; la CNN cataloga 3.567.",
                      f"CNN: corrida del {fecha(c['inicio_corrida']) if c.get('inicio_corrida') else '—'}.",
                  ], chicos=True))
    ), "dos"))

    # 19. Generalización e interpretabilidad
    d.append(("Comparación · generalización e interpretabilidad", "El MLP memoriza más y se explica sobre descriptores", (
        ctx.fig("generalizacion_mlp_cnn.png", "Accuracy por tipo de imagen del MLP y de la CNN")
        + columna(tabla(["Indicador", "MLP", "CNN"], [
            ["Brecha del modelo base", "0,397", "0,142"],
            ["Brecha del modelo final", "0,337", "0,033"],
            ["Caída en alteradas (pts)", "9,0", "1,2"],
            ["Ventaja de las parecidas (pts)", "7,8", "6,1"],
            ["Errores confiados / 1.000", "56,7", "19,4"],
        ], numericas=(1, 2), densa=True), lista([
            "<strong>SHAP (MLP):</strong> qué canal de color y qué zona pesan; exacto, pero sin forma ni textura.",
            "<strong>Grad-CAM (CNN):</strong> dónde mira la red, sobre la misma imagen; se verifica a simple vista.",
        ], chicos=True))
    ), "dos mitad"))

    # 20. Arquitectura seleccionada
    d.append(("Arquitectura seleccionada", "La CNN; el MLP queda como línea base", (
        tabla(["Criterio", "A favor de"], [
            ["Desempeño (F1 +19,8, accuracy +22,0 puntos)", "CNN"],
            ["Generalización (brecha, alteradas, errores confiados)", "CNN"],
            ["Automatización (71,6 % frente a 14,3 % sin revisión)", "CNN"],
            ["Sesgo contra la familia Simpson (0,5 frente a 11,3 puntos)", "CNN"],
            ["Interpretabilidad (Grad-CAM sobre la imagen)", "CNN"],
            ["Catalogación de extremo a extremo (3.567 frente a 1.360 img/s)", "CNN"],
            ["Calibración (ECE 0,044 frente a 0,078)", "MLP"],
            ["Costo de entrenamiento (2,25 frente a 51,7 s por época)", "MLP"],
        ], densa=True)
        + columna(lista([
            "El costo de entrenamiento de la CNN se paga una sola vez: unos 50 minutos por configuración, frente a "
            "unos 3 del MLP.",
            "La mejor calibración del MLP no compensa que se equivoque en el 44 % del test.",
            "El MLP solo convendría sin GPU para entrenar y con personajes de paleta única.",
        ], chicos=True))
    ), "dos"))

    # 21. Ética
    d.append(("Impacto ético", "Un asistente con revisión casi completa, no un juez", lista([
        "<strong>Sesgo contra la familia Simpson:</strong> 46 % de acierto frente a 58 % del resto, justo en los "
        "personajes más buscados. Hay que comunicar la cifra por subgrupo, no solo la global.",
        "<strong>Sesgo por parecido:</strong> para medir presencia en pantalla contaría a Judge Snyder 1,38 veces sus "
        "apariciones y a Maggie 0,61: el error llegaría a decisiones de licencias.",
        "<strong>Revisión humana:</strong> se equivoca en el 44,4 % del test y siempre responde algún personaje. Solo "
        "el 14,3 % supera el umbral de 0,958; el resto, con las tres sugerencias y revisión.",
        "<strong>Origen, ruido en las etiquetas y derechos de autor:</strong> 0,63 en Kaggle y 0,51 en fotogramas; "
        "limpiar las carpetas antes de medir; uso académico, sin versionar el dataset ni los modelos.",
        "<strong>Fuera del dominio:</strong> con fotos de personas sería reconocimiento de personas. No debe usarse así.",
    ]), ""))

    # 22. Limitaciones y propuestas
    d.append(("Limitaciones y propuestas", "Lo que queda por resolver", tabla(["Limitación", "Medida propuesta"], [
        ["No ve la forma: confunde paletas parecidas", "Una red que aprenda sus características (la CNN) o transfer learning"],
        ["Los histogramas cuentan el fondo", "Descriptores sobre la caja de cada personaje (detectar_cajas.py)"],
        ["Sensible al color: −9,0 puntos en alteradas", "Más variantes de color; descriptores normalizados por imagen"],
        ["35 de 36 configuraciones llegaron al tope", "Más épocas para la búsqueda"],
        ["Automatiza solo el 14,3 % al 90 %", "Asistente con tres sugerencias; para automatizar, la CNN"],
        ["Una sola semilla: el podio queda a 0,2 puntos", "Varias semillas o validación cruzada"],
        ["Tomas casi idénticas (+7,8 puntos)", "Partición por grupo de tomas o por episodio"],
    ], densa=False), ""))

    # 23. Conclusiones
    d.append(("Conclusiones", "Qué deja la EP3", lista([
        f"El MLP llega a <strong>{num(j['accuracy_test'])} de accuracy y {num(j['f1_macro_test'])} de F1 macro</strong> "
        "en test y cumple los cinco KPIs, pero la CNN lo supera en 22 y 20 puntos con las mismas imágenes.",
        "<strong>La representación pesa más que la capacidad:</strong> solo color suma 6,9 puntos; una tercera capa, "
        "1,2.",
        "<strong>Un factor a la vez no ve las interacciones:</strong> la grilla bajó el dropout a 0,2 y sumó 5,8 "
        "puntos sobre C1.",
        "<strong>La regularización que sirve cambia lo que la red ve:</strong> dropout, suavizado y alteradas; L2 no.",
        "<strong>La auditoría encuentra lo que la accuracy esconde:</strong> 11,3 puntos menos en la familia Simpson y "
        "solo el 14,3 % catalogable sin revisión al 90 %.",
    ]), ""))

    return d
