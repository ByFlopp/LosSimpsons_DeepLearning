"""Contenido de la presentación de resultados de la CNN (Evaluación Parcial 3).

Las cifras principales salen del resumen de la corrida (`ctx.j`) y del resumen del MLP (`ctx.otra`); las de los
experimentos, que no están en el resumen, son las de las salidas guardadas del notebook de la misma corrida.
"""

from generar_presentacion import cifras, columna, duracion, entero, estado, fecha, leyenda, lista, num, pct, \
    puntos_de, tabla

TITULO = "¿Quién sale en la imagen? — CNN sobre 64 personajes de Los Simpson (EP3)"
DESCRIPCION = ("Resultados de la red convolucional optimizada y auditada: regularización, Random Search, "
               "auditoría de sesgos y comparación con el MLP.")
BARRA = "CNN sobre 64 personajes de Los Simpson · EP3"
PIE = "CNN · Los Simpson · Evaluación Parcial 3"
SUBTITULO_PORTADA = ("Una red convolucional optimizada y auditada para reconocer a 64 personajes, comparada con un "
                     "Perceptrón Multicapa entrenado con las mismas imágenes. Corrida del")


def fecha_portada(ctx):
    """Fecha de la corrida que muestra la portada."""
    return fecha(ctx.j["inicio_corrida"])


def final(ctx):
    """Nota al pie de la página, fuera de las diapositivas."""
    return ("Cifras de <span class=\"dato\">CNN/resultados/newDataset/GPU/rendimiento_gpu_128x128.json</span> y de las "
            "salidas de <span class=\"dato\">CNN/notebooks/LosSimpsons_GPU_4060M.ipynb</span>, corrida del "
            f"{fecha(ctx.j['inicio_corrida'])} ({duracion(ctx.j['duracion_corrida'])}). Las figuras las guarda el "
            "notebook en <span class=\"dato\">CNN/docs/newDataset/figuras/</span>. Generada con "
            "<span class=\"dato\">scripts/newDataset/generar_presentacion.py --modelo cnn</span>.")


def _mlp(ctx, clave, formato=num):
    """Cifra del MLP o «no medida» si su resumen no la tiene."""
    if not ctx.otra or ctx.otra.get(clave) is None:
        return "no medida"
    return formato(ctx.otra[clave])


def diapositivas(ctx):
    """Lista de (sección, título, cuerpo, disposición) de la presentación de la CNN."""
    j = ctx.j
    m = ctx.otra or {}
    d = []

    # 2. Problema
    d.append(("Problema", "Catalogar un archivo que crece más rápido que el equipo", columna(
        cifras([
            ("Personajes", entero(j["personajes"]), "una carpeta por personaje"),
            ("Imágenes originales", "56.268", "60,6 % fotogramas de episodio"),
            ("Partición", "70 / 20 / 10", "estratificada, semilla 42"),
            ("Resolución", j["resolucion"].replace("x", " × "), "la imagen completa, sin aplanar"),
        ]),
        lista([
            "Clasificación multiclase de etiqueta única: el modelo asigna <strong>un personaje</strong> a cada imagen, "
            "con su confianza.",
            "Uso: el modelo propone y una persona valida; el trabajo pasa de <em>etiquetar</em> a <em>revisar</em>.",
            "Acierto estricto: solo cuenta nombrar al personaje de la carpeta. La accuracy por escena es solo una "
            "referencia.",
            "La EP3 optimiza y audita el pipeline de la EP2: el mismo para la CNN y el MLP, con el mismo presupuesto "
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
                "pruebas, y las importan los notebooks de la CNN y del MLP: solo cambia la función que construye el "
                "modelo. La partición va antes que los píxeles, y cada configuración guarda su historial y se reanuda "
                "si la corrida se corta."),
    ), ""))

    # 5. Modelo base
    d.append(("Modelo base", "Cuatro bloques convolucionales y pooling global", (
        tabla(["Componente", "Configuración"], [
            ["Entrada", "128 × 128 × 3 en uint8, escalada dentro del modelo"],
            ["Bloques", "32, 64, 128 y 256 filtros de 3 × 3, BatchNorm, ReLU y max pooling"],
            ["Cabeza", "Pooling global, densa de 256, dropout 0,3, softmax de 64"],
            ["Entrenamiento", "Adam 10⁻³, batch 128, suavizado 0,1, pesos por clase (raíz)"],
        ])
        + columna(cifras([
            ("Parámetros", entero(j["parametros_modelo_final"]), "62,5 % en conv_4"),
            ("Primera convolución", "864", "pesos, frente a 25,2 millones de una densa de 512 sobre los píxeles"),
        ], 2), lista(["El último bloque ve zonas de 38 × 38 píxeles: partes enteras del personaje."], chicos=True))
    ), "dos"))

    # 6. Entrenamiento
    d.append(("Entrenamiento y parada temprana", "La base para sola en la época 42 de 60", (
        ctx.fig("curvas_modelo_base.png", "Curvas de pérdida y accuracy del modelo base")
        + columna(cifras([
            ("Mejor época", "30", "val_loss 1,050"),
            ("Brecha train − val", num(j["brecha_train_val_base"]), "0,877 frente a 0,736"),
            ("Épocas ahorradas", "18", "unos 17 minutos"),
            ("Sin parada temprana", "0,192", "brecha en la última época"),
        ], 2), lista(["La parada temprana devuelve los pesos de la época 30: después, la red solo memorizaba."],
                     chicos=True))
    ), "dos figura-ancha"))

    # 7. Regularización
    d.append(("Regularización", "Lo que sirve es dar variedad a los datos", (
        tabla(["Técnica", "Experimento", "Δ F1 val.", "Brecha"], [
            ["Dropout", "G1 · sin dropout", "−5,7", "0,310"],
            ["Dropout", "E6 · dropout 0,5", "−0,2", "0,130"],
            ["Batch Normalization", "G2 · sin BatchNorm", "−2,5", "0,133"],
            ["L2", "G3 · L2 5·10⁻⁴", "−0,1", "0,194"],
            ["Aumento fuera de línea", "A1 · sin alteradas", "+0,1", "0,182"],
            ["<strong>Aumento en línea</strong>", "<strong>G4</strong>", "<strong>+3,8</strong>", "<strong>0,033</strong>"],
            ["Suavizado", "S1 · sin suavizado", "−1,1", "0,098"],
            ["Combinación", "C1 · G4 + P2 + A1", "+1,8", "0,041"],
        ], numericas=(2, 3), densa=True)
        + columna(lista([
            "Base: F1 macro de validación 0,714, brecha 0,142, <strong>mismo presupuesto para todas</strong>.",
            "El dropout es imprescindible y Batch Normalization aporta 2,5 puntos, aunque duplica el tiempo por época.",
            "L2 y más dropout no cambian nada: restringir los pesos no ayuda.",
            "El aumento en línea es la mejora clara, y las mejoras no se suman: C1 rinde la mitad que G4.",
        ], chicos=True))
    ), "dos"))

    # 8. Un factor a la vez
    d.append(("Optimización · un factor a la vez", "Las decisiones propias de una CNN se notan", (
        ctx.fig("experimentos.png", "F1 macro y accuracy de validación de cada configuración")
        + columna(lista([
            "Sin color (R1) −8,1 puntos de F1; aplanar en lugar de pooling global (E7) −6,1 con 10 veces los "
            "parámetros; un bloque menos (E1) −2,3.",
            "Las peores: tasa de 10⁻⁴ (E4, −10,9) y tanh (E3, −9,2), las dos todavía aprendiendo en la época 60.",
            "E2 (dos convoluciones por bloque) y E5 (batch 512) no caben en los 8 GB de la GPU.",
        ], chicos=True))
    ), "dos figura-ancha"))

    # 9. Random Search
    d.append(("Optimización · Random Search", "Dos etapas sobre 72 combinaciones", (
        ctx.fig("busqueda.png", "F1 macro de validación del Random Search en sus dos etapas")
        + columna(lista([
            "Tasa {10⁻³, 5·10⁻⁴, 2·10⁻⁴} × dropout {0,2 a 0,5} × L2 {0, 10⁻⁴, 5·10⁻⁴} × aumento en línea {no, sí}.",
            "<strong>Etapa 1:</strong> 8 configuraciones al azar con 15 épocas. Ganan las dos con tasa 10⁻³.",
            "<strong>Etapa 2:</strong> las 2 mejores con 60 épocas. RS1, con aumento en línea, llega a 0,751; RS2, sin "
            "él, a 0,711.",
            "Llega a lo mismo que un factor a la vez: el aumento en línea es lo que importa. Costo: 3 h 11 min.",
        ], chicos=True))
    ), "dos figura-ancha"))

    # 10. Selección
    d.append(("Selección del modelo", "Gana G4: la base con aumento de datos en línea", (
        tabla(["Configuración", "F1 macro val.", "Accuracy val.", "Brecha", "Parada"], [
            ["<strong>G4 · aumento en línea</strong>", "<strong>0,752</strong>", "0,770", "0,033", "tope"],
            ["RS1 · Random Search", "0,751", "0,767", "0,057", "tope"],
            ["C1 · combinación", "0,732", "0,749", "0,041", "tope"],
            ["P2 · pesos balanceados", "0,719", "0,739", "0,174", "paciencia"],
            ["Base", "0,714", "0,736", "0,142", "paciencia"],
            ["RS2 · Random Search", "0,711", "0,730", "0,173", "paciencia"],
        ], numericas=(1, 2, 3), densa=True)
        + columna(lista([
            "Se elige por <strong>F1 macro de validación</strong>: promedia los 64 personajes sin ponderar.",
            "Con RS1 es un empate (0,1 puntos); G4 tiene menor pérdida, menor brecha y cambia un solo factor.",
            "+3,8 puntos sobre la base: el doble de la variación entre corridas, y G4 ya había ganado el 13-11.",
            "Reserva: G4, RS1 y C1 llegaron al tope con su mejor época al final; todavía podían mejorar.",
        ], chicos=True))
    ), "dos"))

    # 11. Test y KPIs
    kpis = [
        ("Cobertura de asistencia", "Top-3", "≥ 0,70", num(j["top3_test"]), j["top3_test"] >= 0.70),
        ("Equidad entre personajes", "F1 macro", "≥ 0,45", num(j["f1_macro_test"]), j["f1_macro_test"] >= 0.45),
        ("Automatización sin revisión", "Accuracy", "≥ 0,50", num(j["accuracy_test"]), j["accuracy_test"] >= 0.50),
        ("Ventaja sobre lo trivial", "Accuracy / mayoritaria", "≥ 4×", "13,2×", True),
        ("Costo de catalogación", "Imágenes por segundo", "≥ 1.000", "3.567", True),
    ]
    d.append(("Evaluación en test", "Los cinco KPIs se cumplen", columna(
        cifras([
            ("Accuracy", num(j["accuracy_test"]), "49,6 veces el azar"),
            ("F1 macro", num(j["f1_macro_test"]), f"precision {num(j['precision_macro_test'])} · recall "
                                                  f"{num(j['recall_macro_test'])}"),
            ("Top-3", num(j["top3_test"]), "la respuesta entre tres sugerencias"),
            ("ROC-AUC macro", num(j["roc_auc_macro_test"]), "ningún personaje bajo 0,90"),
        ]),
        tabla(["KPI", "Métrica", "Umbral", "CNN", ""], [[a, b, c, v, estado(ok)] for a, b, c, v, ok in kpis],
              numericas=(2, 3), densa=True),
    ), ""))

    # 12. ROC y personajes
    d.append(("Evaluación por personaje", "Falla con quien se parece a otro, no con quien tiene pocas imágenes", (
        ctx.fig("curvas_roc.png", "Curvas ROC de los personajes mejor y peor separados")
        + columna(lista([
            "53 de 64 personajes con F1 ≥ 0,60 y ninguno bajo 0,30. Los mejores: Kang y Kodos (0,96), Krusty (0,94), "
            "Kent Brockman (0,93).",
            "Los peores: Rod Flanders (0,32), Helen Lovejoy (0,39), Louie (0,44), Maude (0,45) y Todd (0,46).",
            "Maggie, con 135 imágenes de entrenamiento, llega a 0,77: la correlación entre imágenes y F1 es de "
            "solo 0,25.",
            "Rod tiene ROC-AUC de 0,95 y F1 de 0,32: casi siempre es candidato, pero Todd queda por delante.",
        ], chicos=True))
    ), "dos mitad"))

    # 13. Confusiones y errores
    d.append(("Análisis de errores", "Personajes casi iguales o que salen juntos", (
        tabla(["Real", "Predicho", "% de la clase"], [
            ["Selma Bouvier", "Patty Bouvier", "42,2"],
            ["Todd Flanders", "Rod Flanders", "30,0"],
            ["Louie", "Fat Tony", "22,4"],
            ["Lenny Leonard", "Carl Carlson", "20,7"],
            ["Chief Wiggum", "Lou", "19,8"],
            ["Charles Montgomery Burns", "Waylon Smithers", "11,9"],
        ], numericas=(2,), densa=True)
        + columna(cifras([
            ("Errores", entero(1589), "22,5 % del test"),
            ("Nombran a otro presente", "513", "32,3 % de los errores"),
            ("Confiados (> 0,80)", entero(j["errores_confiados_test"]), "261 en el MLP"),
        ], 3), lista([
            "Ninguna de las quince confusiones más frecuentes es de la familia Simpson.",
            "Los diez errores más confiados nombran a un personaje que está en la imagen; algunos muestran ruido en "
            "las etiquetas (Agnes Skinner en la carpeta de Helen Lovejoy).",
            f"Accuracy por escena, de referencia: {num(j['accuracy_por_escena_test'])}. La accuracy es una cota "
            "inferior.",
        ], chicos=True))
    ), "dos"))

    # 14. Mapas de activación
    d.append(("Explicabilidad", "Del color compartido al patrón de cada personaje", (
        ctx.fig("mapas_activacion_cuatro_imagenes.jpg", "Mapas de activación de Homero, Marge, Bart y Lisa")
        + columna(lista([
            "Con Homero, Marge, Bart y Lisa, la correlación entre los filtros activados es de 0,40 en el tercer bloque "
            "—lo común a los Simpson— y cae a 0,14 en el cuarto: cada personaje enciende su propio conjunto.",
            "La representación se vuelve selectiva: 38 % de valores activos en el primer bloque, 1 % en el último.",
        ], chicos=True))
    ), "dos figura-ancha"))

    # 15. Grad-CAM
    d.append(("Explicabilidad · Grad-CAM", "La red mira al personaje", (
        ctx.fig("mapa_activacion_cada_clase.jpg", "Grad-CAM de un ejemplo de cada personaje", "lamina alta")
        + columna(cifras([
            ("10 % más cálido", "72,7 %", "de la influencia (10 % si mirara todo igual)"),
            ("Mitad central", "49,6 %", "en los aciertos (25 % uniforme)"),
        ], 2), lista([
            "En los 64 personajes el calor cae sobre el personaje, casi siempre en la cara.",
            "Concentrar el calor acompaña al reconocimiento: correlación de 0,52 con el recall.",
            "Los de calor más disperso —Louie, Rod, el Rico Texano, Helen— están entre los de peor F1.",
            "Los errores más confiados miran la cara de otro personaje presente.",
        ], chicos=True))
    ), "dos mitad"))

    # 16. Calibración y umbral
    d.append(("Auditoría · confiabilidad", "Subconfiado, y con un umbral que se traslada a test", (
        ctx.fig("auditoria_2.png", "Diagrama de confiabilidad y distribución de la confianza")
        + columna(cifras([
            ("ECE", num(j["ece_test"]), "confianza 0,698 frente a accuracy 0,775"),
            ("Umbral de revisión", num(j["umbral_revision"]), "elegido en validación para un 90 %"),
            ("Resuelve solo", pct(j["cobertura_umbral_test"]), "del test"),
            ("Acierto de lo aceptado", pct(j["accuracy_umbral_test"]), "el resto va a revisión humana"),
        ], 2))
    ), "dos figura-ancha"))

    # 17. Sesgos
    d.append(("Auditoría · sesgos por subgrupo", "El origen pesa más que el desbalance", columna(
        ctx.fig("auditoria_3.png", "Accuracy de la CNN por subgrupo"),
        tabla(["Variable", "Grupos", "Accuracy", "Brecha"], [
            ["Origen", "Kaggle / fotograma de episodio",
             f"{num(j['accuracy_kaggle_test'])} / {num(j['accuracy_episodio_test'])}",
             "26,3 pts"],
            ["Tamaño de la clase", "grandes / medianas / pequeñas", "0,795 / 0,768 / 0,727", "6,8 pts"],
            ["Familia Simpson", "familia / resto",
             f"{num(j['accuracy_familia_simpson_test'])} / {num(j['accuracy_resto_personajes_test'])}", "0,5 pts"],
        ], numericas=(2, 3), densa=True),
        leyenda("Los fotogramas, con planos generales y varios personajes, son el 60 % del test: la cifra que "
                "representa un archivo de episodios es 0,670. Elegir fotogramas con la CNN de octubre (Frinkiac) no "
                "los volvió fáciles."),
    ), ""))

    # 18. Comparación con el MLP
    filas = [
        ["Configuración final", m.get("configuracion_final", "—"), j["configuracion_final"]],
        ["Parámetros", _mlp(ctx, "parametros_modelo_final", entero), entero(j["parametros_modelo_final"])],
        ["Accuracy", _mlp(ctx, "accuracy_test"), f"<strong>{num(j['accuracy_test'])}</strong>"],
        ["F1 macro", _mlp(ctx, "f1_macro_test"), f"<strong>{num(j['f1_macro_test'])}</strong>"],
        ["Top-3", _mlp(ctx, "top3_test"), num(j["top3_test"])],
        ["Accuracy en alteradas", _mlp(ctx, "accuracy_alteradas_test"), num(j["accuracy_alteradas_test"])],
        ["Errores confiados", _mlp(ctx, "errores_confiados_test", entero), entero(j["errores_confiados_test"])],
        ["Segundos por época", _mlp(ctx, "benchmark_segundos_por_epoca", lambda v: num(v, 1)),
         num(j["benchmark_segundos_por_epoca"], 1)],
        ["Inferencia de la red (img/s)", _mlp(ctx, "inferencia_imagenes_por_segundo", entero),
         entero(j["inferencia_imagenes_por_segundo"])],
    ]
    diferencia = puntos_de(j["f1_macro_test"] - m["f1_macro_test"]) if m.get("f1_macro_test") else "—"
    d.append(("Comparación con el MLP", "Mismos datos, mismo proceso: solo cambia el modelo", (
        tabla(["Concepto", "MLP", "CNN"], filas, numericas=(1, 2), densa=True)
        + columna(cifras([("F1 macro", diferencia, "puntos a favor de la CNN"),
                          ("Por época", "×24,6", "más cara la CNN")], 2),
                  lista([
                      "El MLP recibe 370 descriptores de color: confunde a los personajes que comparten paleta.",
                      "La red del MLP infiere más rápido, pero sus descriptores en la CPU lo limitan a unas 430 "
                      "imágenes por segundo de extremo a extremo; la CNN cataloga 3.567.",
                      f"MLP: corrida del {fecha(m['inicio_corrida']) if m.get('inicio_corrida') else '—'}.",
                  ], chicos=True))
    ), "dos"))

    # 19. Generalización e interpretabilidad
    d.append(("Comparación · generalización e interpretabilidad", "La CNN memoriza menos y se explica sobre la imagen", (
        ctx.fig("generalizacion_mlp_cnn.png", "Accuracy por tipo de imagen del MLP y de la CNN")
        + columna(tabla(["Indicador", "MLP", "CNN"], [
            ["Brecha del modelo base", "0,378", "0,142"],
            ["Caída en alteradas (pts)", "8,7", "1,2"],
            ["Ventaja de las parecidas (pts)", "6,3", "6,1"],
            ["Errores confiados / 1.000", "37,0", "19,4"],
        ], numericas=(1, 2), densa=True), lista([
            "<strong>Grad-CAM (CNN):</strong> dónde mira la red, sobre la misma imagen; se verifica a simple vista.",
            "<strong>SHAP (MLP):</strong> qué canal de color y qué zona pesan; exacto, pero sin forma ni textura.",
        ], chicos=True))
    ), "dos mitad"))

    # 20. Arquitectura seleccionada
    d.append(("Arquitectura seleccionada", "La CNN, en su configuración G4", (
        tabla(["Criterio", "A favor de"], [
            ["Desempeño (F1 +20, accuracy +23 puntos)", "CNN"],
            ["Generalización (brecha, alteradas, errores confiados)", "CNN"],
            ["Interpretabilidad (Grad-CAM sobre la imagen)", "CNN"],
            ["Catalogación de extremo a extremo (3.567 frente a 430 img/s)", "CNN"],
            ["Parámetros (473.120 frente a 1.059.648)", "CNN"],
            ["Costo de entrenamiento (51,7 frente a 2,1 s por época)", "MLP"],
        ], densa=True)
        + columna(lista([
            "Gana en todo menos en el costo de entrenamiento, que se paga una sola vez: unos 50 minutos por "
            "configuración en la RTX 4060.",
            "La calidad y la velocidad de catalogación se aprovechan en cada imagen del archivo.",
            "El MLP solo convendría sin GPU para entrenar y con personajes de paleta única.",
        ]))
    ), "dos"))

    # 21. Ética
    d.append(("Impacto ético", "Un asistente con revisión humana, no un juez", lista([
        "<strong>Sesgo de origen:</strong> 0,93 en Kaggle y 0,67 en fotogramas. Para un archivo de episodios hay que "
        "comunicar la cifra de los fotogramas, no la global.",
        "<strong>Sesgo por parecido:</strong> para medir presencia en pantalla contaría a Patty y a Lou 1,41 veces sus "
        "apariciones y a Selma 0,80: el error llegaría a decisiones de licencias.",
        "<strong>Revisión humana:</strong> se equivoca en el 22,5 % del test y siempre responde algún personaje. Aceptar "
        "solo lo que supera el umbral (71,6 % del test, 90,4 % de acierto) y revisar el resto.",
        "<strong>Ruido en las etiquetas</strong> y <strong>derechos de autor</strong>: limpiar las carpetas antes de "
        "medir; uso académico, sin versionar el dataset ni los modelos.",
        "<strong>Fuera del dominio:</strong> con fotos de personas sería reconocimiento facial. No debe usarse así.",
    ]), ""))

    # 22. Limitaciones y propuestas
    d.append(("Limitaciones y propuestas", "Lo que queda por resolver", tabla(["Limitación", "Medida propuesta"], [
        ["Sesgo de origen (0,67 en fotogramas)", "Más fotogramas difíciles, sin filtrarlos con un modelo anterior"],
        ["G4, RS1 y C1 todavía mejoraban en la época 60", "Más épocas para las configuraciones con aumento en línea"],
        ["Personajes parecidos o que salen juntos", "Una caja por personaje (detectar_cajas.py) o salida multietiqueta"],
        ["Subconfianza (ECE 0,078)", "Calibración por temperatura en validación"],
        ["L2 sin efecto", "Weight decay desacoplado (AdamW) y SpatialDropout2D"],
        ["Entrenada desde cero", "Transfer learning (MobileNet, ResNet, EfficientNet)"],
        ["Variación entre corridas (hasta 1,9 puntos)", "Varias semillas o validación cruzada"],
        ["Tomas casi idénticas (+6,1 puntos)", "Partición por grupo de tomas o por episodio"],
    ], densa=False), ""))

    # 23. Conclusiones
    d.append(("Conclusiones", "Qué deja la EP3", lista([
        f"La CNN llega a <strong>{num(j['accuracy_test'])} de accuracy y {num(j['f1_macro_test'])} de F1 macro</strong> "
        "en test, 23 y 20 puntos sobre el MLP con las mismas imágenes, y cumple los cinco KPIs.",
        "<strong>La regularización que sirve es la que da variedad:</strong> el aumento en línea (+3,8 puntos, brecha "
        "0,033); el Random Search llega a lo mismo.",
        "<strong>El mismo presupuesto cambia la lectura:</strong> con 60 épocas para todas, la ventaja de G4 se duplica.",
        "<strong>Lo difícil es parecerse a otro,</strong> no tener pocas imágenes, y Grad-CAM muestra que la red mira "
        "al personaje.",
        "<strong>La auditoría encuentra lo que la accuracy esconde:</strong> el sesgo de origen, la subconfianza y un "
        "umbral que cataloga sin revisión tres de cada cuatro imágenes al 90 %.",
    ]), ""))

    return d
