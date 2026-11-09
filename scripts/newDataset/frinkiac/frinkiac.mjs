#!/usr/bin/env node
// Lanzador de `pnpm frinkiac`.
//
//   pnpm frinkiac                     todo el flujo (guiones, inventario, revisiones, recolección)
//   pnpm frinkiac --sin-issue         lo mismo, sin publicar el issue en GitHub
//   pnpm frinkiac estado              avance por personaje
//   pnpm frinkiac exportar            copia las aprobadas de final/ al dataset
//   pnpm frinkiac limpiar             borra lo generado (no el dataset) para empezar de cero
//   pnpm frinkiac clasificar          la CNN separa por_revisar en seguras, dudosas y ruido (usa .venv-gpu)
//   pnpm frinkiac <subcomando> ...    cualquier subcomando de ampliar_con_frinkiac.py
//   pnpm frinkiac pruebas             pruebas automáticas
//
// La primera vez crea el entorno .venv-frinkiac en la raíz del repo e instala las dependencias; las
// reinstala si cambian los requirements. Si la recolección se cae, la vuelve a lanzar (como systemd en
// la EC2): cada arranque retoma desde donde quedó. Ctrl+C la detiene de forma ordenada.

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const aqui = dirname(fileURLToPath(import.meta.url));
const raiz = resolve(aqui, "..", "..", "..");
const venv = join(raiz, ".venv-frinkiac");
const esWindows = process.platform === "win32";
const pythonVenv = join(venv, esWindows ? join("Scripts", "python.exe") : join("bin", "python"));
const script = join(aqui, "ampliar_con_frinkiac.py");
// La CNN corre en el entorno del proyecto (Keras sobre PyTorch con CUDA), no en .venv-frinkiac
const pythonGpu = join(raiz, ".venv-gpu", esWindows ? join("Scripts", "python.exe") : join("bin", "python"));
const SUBCOMANDOS = new Set(["flujo", "inventario", "mapear", "recolectar", "estado", "revisar", "aplicar-revision",
  "exportar", "limpiar"]);
// Códigos de ampliar_con_frinkiac.py que no se reintentan: terminó, falta configurar algo, detenido a
// pedido, otro proceso está recolectando
const NO_REINTENTAR = new Set([0, 2, 3, 5]);
const ESPERA_REINICIO_S = 30;

function pythonBase() {
  const candidatos = process.env.FRINKIAC_PYTHON
    ? [[process.env.FRINKIAC_PYTHON]]
    : esWindows ? [["py", "-3"], ["python"]] : [["python3"], ["python"]];
  for (const [comando, ...previos] of candidatos) {
    const prueba = spawnSync(comando, [...previos, "-c", "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"]);
    if (prueba.status === 0) return [comando, previos];
  }
  console.error("Se necesita Python 3.11 o superior (o define FRINKIAC_PYTHON con su ruta).");
  process.exit(2);
}

function huella(archivos) {
  const hash = createHash("sha256");
  for (const archivo of archivos) hash.update(readFileSync(archivo));
  return hash.digest("hex");
}

function asegurarEntorno(requisitos) {
  const archivos = [join(aqui, "requirements-frinkiac.txt"), ...(requisitos.endsWith("pruebas.txt") ? [requisitos] : [])];
  const marca = join(venv, `.instalado-${requisitos.endsWith("pruebas.txt") ? "pruebas" : "base"}`);
  if (!existsSync(pythonVenv)) {
    const [comando, previos] = pythonBase();
    console.log(`Creando el entorno ${venv} ...`);
    const creado = spawnSync(comando, [...previos, "-m", "venv", venv], { stdio: "inherit" });
    if (creado.status !== 0) process.exit(2);
  }
  const actual = huella(archivos);
  if (!existsSync(marca) || readFileSync(marca, "utf8") !== actual) {
    console.log("Instalando dependencias de Python ...");
    const pip = spawnSync(pythonVenv, ["-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", requisitos],
      { stdio: "inherit" });
    if (pip.status !== 0) process.exit(2);
    writeFileSync(marca, actual);
  }
}

let hijo = null;
let detenido = false;
// Ctrl+C llega también al proceso de Python, que termina la imagen en curso y sale solo: aquí solo se
// evita relanzarlo
process.on("SIGINT", () => {
  detenido = true;
  if (!hijo) process.exit(3);
});

function ejecutar(argumentos, python = pythonVenv) {
  return new Promise((listo) => {
    hijo = spawn(python, argumentos, {
      cwd: raiz,
      stdio: "inherit",
      env: { ...process.env, PYTHONUTF8: "1", PYTHONUNBUFFERED: "1" },
    });
    hijo.on("exit", (codigo, senal) => {
      hijo = null;
      listo(codigo ?? (senal ? 3 : 1));
    });
  });
}

const dormir = (segundos) => new Promise((listo) => setTimeout(listo, segundos * 1000));

async function principal() {
  const argumentos = process.argv.slice(2).filter((a) => a !== "--");
  if (argumentos[0] === "pruebas") {
    asegurarEntorno(join(aqui, "requirements-frinkiac-pruebas.txt"));
    process.exit(await ejecutar(["-m", "pytest", join(aqui, "tests"), "-q", "-p", "no:cacheprovider",
      ...argumentos.slice(1)]));
  }
  if (argumentos[0] === "clasificar") {
    if (!existsSync(pythonGpu)) {
      console.error(`No se encontró ${pythonGpu}. La clasificación usa el entorno de la CNN: créalo con `
        + "requirements-gpu-windows.txt (pasos en PMC/notebooks/oldDataset/README_GPU.md).");
      process.exit(2);
    }
    process.exit(await ejecutar([join(aqui, "clasificar_con_cnn.py"), ...argumentos.slice(1)], pythonGpu));
  }
  asegurarEntorno(join(aqui, "requirements-frinkiac.txt"));
  const subcomando = SUBCOMANDOS.has(argumentos[0]) ? argumentos[0] : "flujo";
  const resto = SUBCOMANDOS.has(argumentos[0]) ? argumentos.slice(1) : argumentos;
  const comando = [script, "--log-formato", "texto", subcomando, ...resto];
  const reintentable = subcomando === "flujo" || subcomando === "recolectar";
  for (let intento = 1; ; intento++) {
    const codigo = await ejecutar(comando);
    if (!reintentable || NO_REINTENTAR.has(codigo) || detenido) process.exit(detenido && codigo !== 0 ? 3 : codigo);
    console.log(`\nLa recolección salió con código ${codigo}; se relanza en ${ESPERA_REINICIO_S} s `
      + `(reintento ${intento}; Ctrl+C para cancelar).`);
    await dormir(ESPERA_REINICIO_S);
    if (detenido) process.exit(3);
  }
}

principal();
