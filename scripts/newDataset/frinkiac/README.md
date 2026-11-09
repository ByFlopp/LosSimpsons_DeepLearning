# Ampliación del dataset con Frinkiac

Este flujo completa **1000 imágenes originales por personaje** (las que ya hay más las nuevas) con fotogramas de [Frinkiac](https://frinkiac.com). Afecta a 52 de los 64 personajes, que hoy tienen menos de 1000 originales: desde `louie`, con 94, hasta `sideshow_bob`, con 960. En total faltan 32.148 imágenes.

**Solo cuentan las originales.** Las variantes `__alterada` y `__parecida` no entran en el conteo, en la comparación de unicidad ni en lo que se guarda. Esas variantes se generan **después**, con [generar_imagenes_alteradas.py](../generar_imagenes_alteradas.py), cuando el dataset ya tiene sus 1000 originales.

## Uso: `pnpm frinkiac`

Desde la raíz del repositorio (requiere Node 18+, pnpm y Python 3.11+):

```bash
pnpm frinkiac                # todo el flujo; se puede detener con Ctrl+C y retomar cuando quieras
pnpm frinkiac --sin-issue    # igual, pero sin publicar el issue en GitHub
pnpm frinkiac estado         # avance por personaje
pnpm frinkiac exportar       # copia las aprobadas al dataset (sin sobrescribir nada)
pnpm frinkiac clasificar     # la CNN separa por_revisar en seguras, dudosas y ruido
pnpm frinkiac:pruebas        # pruebas automáticas
```

La primera vez crea el entorno `.venv-frinkiac` e instala las dependencias. Si la recolección se cae, el lanzador la vuelve a iniciar, y cada arranque retoma desde donde quedó.

Todo queda **junto al dataset, en una carpeta aparte**. El dataset no se toca hasta que ejecutes `exportar`:

```
C:\Users\<usuario>\Desktop\
├── simpsons_dataset\               el dataset
└── frinkiac_ampliacion\
    ├── LEEME.txt
    ├── por_revisar\<personaje>\    AQUÍ SE REVISA
    ├── final\<personaje>\          aprobadas que completan las 1000
    ├── staging\<personaje>\        todas las nuevas, incluidas las rechazadas (nunca se borra nada)
    └── revision\, control\, inputs\, estado\, .datos\   uso interno
```

Para usar otra carpeta, pasa `--local D:\otra\carpeta`. Para cambiar el dataset, define `SIMPSONS_DATASET` o pasa `--dataset`.

### Qué hace cada ejecución

1. **Guiones.** Busca `simpsons_script_lines.csv` y `simpsons_episodes.csv` de [The Simpsons by the Data](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset) en este orden:
   1. sueltos o dentro del `.zip` de Kaggle, en Descargas, el Escritorio, junto al dataset o en la raíz del repo;
   2. en la API de Kaggle, si hay credenciales (`~/.kaggle/kaggle.json`);
   3. como último recurso, en la copia de TidyTuesday, que **solo trae episodios desde 2010**. En ese caso lo avisa.

   Para tener el guion completo, baja el zip de Kaggle a Descargas. La próxima ejecución lo usa y conserva lo ya recolectado.
2. **Inventario.** Cuenta las originales del dataset. Solo lo rehace si el dataset cambió.
3. **Nombres.** Avisa qué personajes no tienen líneas en el guion según [personajes_frinkiac.json](personajes_frinkiac.json). `pnpm frinkiac mapear --guiones <csv>` sugiere nombres.
4. **Revisiones.** Aplica las carpetas de `por_revisar` que ya terminaste de revisar.
5. **Recolección.** Baja fotogramas únicos hasta la meta. Cada personaje aparece en `por_revisar` **apenas termina**, sin esperar a los demás. Al final abre un issue en el repo que menciona a @DiegoPyLL. Para eso usa la sesión de `gh auth login` o `GITHUB_TOKEN`.
6. **Siguiente paso.** Indica qué carpetas quedan por revisar y qué personajes están completos.

### Revisar a mano

En `por_revisar\<personaje>\`:

1. Borra las imágenes en las que el personaje **no** aparece.
2. Cuando termines esa carpeta, borra también `_PENDIENTE.txt`. Mientras ese archivo exista, la carpeta se considera en revisión: no se aplica y pueden llegar más imágenes.
3. Vuelve a ejecutar `pnpm frinkiac`:
   - lo que quedó en la carpeta se aprueba;
   - lo borrado se rechaza y se repone con otros fotogramas, que vuelven a `por_revisar`;
   - al llegar a 1000, el personaje pasa a `final\`.

Si borras todas las imágenes de una carpeta grande, el script lo trata como un posible accidente y no lo aplica hasta que ejecutes `pnpm frinkiac --forzar`.

### Revisar menos: `pnpm frinkiac clasificar`

La CNN final del proyecto ([CNN/](../../../CNN/README.md)) separa las imágenes de `por_revisar` en tres grupos. Solo hay que revisar a mano uno de ellos. Corre en el entorno `.venv-gpu` (en la RTX 4060 tarda unos 2,5 minutos para 27.000 imágenes) y solo clasifica lo nuevo, así que se puede repetir sin problema.

```
frinkiac_ampliacion\clasificacion\
├── seguras\<personaje>\    la CNN reconoce al personaje: se aprueban
├── dudosas\<personaje>\    AQUÍ SE REVISA
├── ruido\<personaje>\      la CNN no lo ve: se rechazan
└── _registro.csv           grupo y probabilidades de cada imagen
```

1. Ejecuta `pnpm frinkiac clasificar`.
2. En `dudosas\<personaje>\`, borra las imágenes en las que el personaje **no** aparece. Al terminar, borra su `_PENDIENTE.txt`.
3. Opcional: si en `ruido\` hay alguna que sí sirve, muévela a `seguras\`. Si en `seguras\` hay un error, bórralo.
4. Ejecuta `pnpm frinkiac`. Por cada personaje listo:
   - lo que quedó en `seguras\` y `dudosas\` se aprueba;
   - lo de `ruido\` y lo borrado se rechaza y se repone, como en la revisión a mano.
5. Las imágenes repuestas llegan a `por_revisar`: vuelve al paso 1.

Las carpetas tienen enlaces duros a las de `por_revisar`, sin copias ni espacio extra. Borrar en ellas no borra nada de `staging\`. Un personaje no se aplica mientras tenga dudosas sin revisar o imágenes nuevas sin clasificar. Si terminas un personaje a mano en `por_revisar` (borrando su `_PENDIENTE.txt`), la clasificación no lo toca.

Cómo decide. Cada fotograma se mira en 4 vistas: completo y recortes cuadrados a la izquierda, al centro y a la derecha. En un plano ancho, el personaje suele ocupar solo una parte. Con *p* la probabilidad que la CNN da al personaje de la carpeta:

| Grupo | Regla | Muestra de 30 revisada a ojo (nov. 2026) |
|---|---|---|
| segura | *p* ≥ 0,80 en el fotograma completo, o ≥ 0,90 en algún recorte | 29 de 30 y 30 de 30 correctas |
| dudosa | el resto | entre 0,2 y 0,8, el personaje aparece en ~25 de 30; entre 0,05 y 0,2, en ~1 de 3 |
| ruido | *p* < 0,05 en las 4 vistas | en ~4 de 30 aparece, casi siempre pequeño o al fondo |

En la primera recolección quedaron 3.876 seguras (14 %), 9.956 dudosas (37 %) y 13.015 de ruido (49 %). Los umbrales se cambian con `--umbral-segura`, `--umbral-recorte` y `--umbral-ruido`. La CNN reconoce poco a algunos personajes (`dolph_starbeam`, `kearney_zzyzwicz`, `maude_flanders`, `rod_flanders`, `todd_flanders`, `helen_lovejoy`): casi no tienen seguras y su revisión recae en las dudosas.

Con todo completo:
1. Ejecuta `pnpm frinkiac exportar`.
2. Ejecuta [generar_imagenes_alteradas.py](../generar_imagenes_alteradas.py) para regenerar las sintéticas.

## Cómo se obtienen las imágenes

1. El guion de Kaggle indica qué personaje dice cada línea, en qué episodio y en qué instante.
2. Frinkiac busca el texto y devuelve el subtítulo exacto. De ese tramo se toman hasta 3 fotogramas separados por al menos 1 s. Si no alcanza, otras dos pasadas abren la ventana ±3 s y ±6 s.
3. Se descarta un fotograma cuando cumple cualquiera de estas condiciones:
   - su episodio e instante ya están en el dataset;
   - su sha256 se repite;
   - se parece demasiado (similitud de miniaturas ≥ 0,90) a otra imagen del mismo personaje. Es la misma regla que `_parecidas`.
4. Que un personaje hable no garantiza que se vea en ese plano. Por eso **toda imagen pasa por revisión**, y se recolecta un 15 % extra (`--margen`) para absorber los rechazos.

## Garantías: nada se pierde ni se repite

| Riesgo | Protección |
|---|---|
| Corte de luz, cierre o Ctrl+C a mitad de una imagen | Una imagen solo cuenta como guardada cuando el almacén confirma su escritura y su sha256. Ctrl+C termina la imagen en curso y guarda el estado. Al volver a ejecutar se retoma. |
| Guardar dos veces la misma imagen | El nombre sale del episodio y del instante (`louie__S05E10__t0123456.jpg`, el formato del dataset). Se crea de forma atómica y **nunca** sobrescribe: en S3 con `If-None-Match: *`; en local con un enlace o rename que falla si el archivo ya existe. |
| Se pierde la base interna | Se reconstruye desde `staging\`, que es la fuente de verdad. Además se respalda cada 5 minutos. |
| Dos ejecuciones a la vez | Un cerrojo por máquina más un candado con lease en el almacén. La segunda se detiene con un aviso. |
| Frinkiac falla o limita | Un máximo de 2 peticiones por segundo y backoff exponencial que respeta `Retry-After`. Tras 20 fallos seguidos, una pausa de 10 minutos. |
| Windows bloquea un archivo (antivirus, indexador) | Las escrituras reintentan unos segundos. Los JSON se leen y escriben bajo cerrojo. |
| El issue se duplica | Un marcador oculto con el id de la corrida: los reinicios comentan en el mismo issue y nunca repiten un aviso. |
| Se aprueban imágenes que nadie vio | Una carpeta solo se aplica cuando borras su `_PENDIENTE.txt`. |

## En una EC2 con S3 (opcional)

El mismo script corre como servicio en AWS, para no depender del PC encendido:

1. Ejecuta `BUCKET=<bucket> REGION=<región> bash scripts/newDataset/frinkiac/ec2/preparar_aws.sh`. Prepara:
   - el bucket, con versionado, sin acceso público, cifrado y con limpieza de versiones antiguas a los 30 días;
   - el token de GitHub en SSM;
   - el rol IAM, con los permisos de [ec2/iam_policy.json](ec2/iam_policy.json).
2. Guarda las entradas en el bucket con `pnpm frinkiac inventario --bucket <bucket> --guiones <csv> --episodios <csv>`.
3. Haz push de `main` y lanza una instancia Amazon Linux 2023 (`t3.small`; spot sirve) con el perfil `frinkiac-ec2` y [ec2/setup_ec2.sh](ec2/setup_ec2.sh) como *user-data*. systemd ([ec2/frinkiac.service](ec2/frinkiac.service)) lo reinicia si se cae y lo detiene limpio ante un aviso de spot.
4. Revisa desde el PC con `pnpm frinkiac revisar --bucket <bucket> --personaje <p>` y aplica con `pnpm frinkiac aplicar-revision --bucket <bucket> --personaje <p>`.

Códigos de salida:

| Código | Significado |
|---|---|
| 0 | Terminó |
| 2 | Falta configurar algo; no se reintenta |
| 3 | Detenido con Ctrl+C o por aviso de spot |
| 4 | Perdió el candado |
| 5 | Otro proceso está recolectando |
| 1 | Error inesperado |

## Límites conocidos

- El guion de Kaggle trae 600 episodios (temporadas 1 a 28). Personajes con pocas líneas habladas, como `maggie_simpson` (7), `disco_stu` (43) o `louie` (103), pueden agotarlas antes de llegar a 1000. Quedan como **agotado**, se informa en el issue y sus aprobadas igual pasan a `final\`.
- Las pasadas de ±3 s y ±6 s traen más planos en los que el personaje no aparece, así que esperen más rechazos.
- A 2 peticiones por segundo, la recolección completa toma del orden de uno o dos días. Se puede pausar y retomar cuando quieras.
