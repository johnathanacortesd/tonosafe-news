# -*- coding: utf-8 -*-
"""Verificacion con TypeSafe (System One) de las etiquetas del motor de Tono/Tema/Sub-tema.

Para que sirve, en una linea: es un SEGUNDO ETIQUETADOR independiente que responde
preguntas tipadas con probabilidad calibrada, no texto. No genera codigo ni redacta
revisiones: devuelve `choice` / `noul` / `score` que ESTE modulo combina con reglas
explicitas. Con eso localizamos donde el motor (gpt-4.1-nano) probablemente se equivoca,
en vez de re-etiquetar todo a ciegas.

Diseño alineado a SPEC_TONO_TEMA.md:
  §0  Sin TYPESAFE_API_KEY: error claro. NUNCA se cae en silencio a heuristicas.
  §1  El estado separa `lo_que_se_dice_de_la_entidad` (decide el tono) de
      `contexto_del_hecho` (sirve para el tema, NO decide el tono).
  §1  Regla 3 determinista en codigo: sin mencion -> Neutro, sin llamar al modelo.
  §2.1 Guarda negativa en codigo: Negativo con `critica_dirigida` baja -> Neutro.
  Tema = lista CERRADA del cliente: cualquier respuesta fuera de la lista se rechaza.

Transporte: HTTP directo con `requests` (ya es dependencia del proyecto, mismo idioma
que analyzer_tono_tema.llamar_llm). Alternativa equivalente: `pip install typesafe-sdk`
y usar TypeSafeClient; el contrato es el mismo.

Credencial (server-side siempre, nunca en el navegador):
  Streamlit Cloud -> Settings -> Secrets:  TYPESAFE_API_KEY = "ts-..."
  Local/CLI       -> variable de entorno TYPESAFE_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODELO_DEFECTO = "jev-latest"
VARIABLE_ENTORNO = "TYPESAFE_API_KEY"

# Umbrales EXPLICITOS. Se calibran con los casos dorados, no se copian de un cookbook.
UMBRAL_MENCION = 0.50      # por debajo: la nota no habla de la entidad -> Neutro (§1 regla 3)
UMBRAL_CRITICA = 0.50      # por debajo: el Negativo no tiene señalamiento dirigido -> Neutro (§2.1)

TEXTO_COMPLETO_MAX = 1200


class TypeSafeSinCredencial(RuntimeError):
    """Falta TYPESAFE_API_KEY: se detiene con aviso, no se degrada a heuristicas (SPEC §0)."""


class TypeSafeError(RuntimeError):
    """La API respondio con un error que no se arregla reintentando (401, 422, ...)."""


# ============================================================================
# 1. Credencial
# ============================================================================
def api_key(explicita: Optional[str] = None) -> str:
    """Busca la key en: argumento -> entorno -> st.secrets. Vacio = no hay credencial."""
    if explicita and str(explicita).strip():
        return str(explicita).strip()
    del_entorno = os.environ.get(VARIABLE_ENTORNO)
    if del_entorno and del_entorno.strip():
        return del_entorno.strip()
    try:  # Streamlit Cloud / local con .streamlit/secrets.toml
        import streamlit as st
        valor = st.secrets.get(VARIABLE_ENTORNO)
        if valor and str(valor).strip():
            return str(valor).strip()
    except Exception:
        pass
    return ""


# ============================================================================
# 2. Constructores de preguntas tipadas
# ============================================================================
def pregunta_noul(instrucciones: str, si: str = "", no: str = "") -> Dict[str, Any]:
    p: Dict[str, Any] = {"type": "noul", "instructions": instrucciones}
    if si or no:
        p["criteria"] = {"true": si or "El enunciado se cumple", "false": no or "El enunciado no se cumple"}
    return p


def pregunta_choice(instrucciones: str, opciones: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """opciones: {valor: rubrica|None}. El valor es el que vuelve en `choice`."""
    return {"type": "choice", "instructions": instrucciones, "criteria": dict(opciones)}


def pregunta_score(instrucciones: str, niveles: Sequence[str]) -> Dict[str, Any]:
    if len(niveles) < 2:
        raise ValueError("Un Score necesita al menos dos niveles (contrato de la API).")
    return {"type": "score", "instructions": instrucciones, "criteria": list(niveles)}


# ============================================================================
# 3. Estado y preguntas por grupo
# ============================================================================
def _contexto_del_grupo(grupo: dict, cfg: dict) -> str:
    """Evidencia de marca del grupo: la que ya calculo el motor si viene, si no se recalcula."""
    if grupo.get("contexto"):
        return str(grupo["contexto"])
    try:
        from analyzer_tono_tema import _contexto_marca
        return _contexto_marca(grupo.get("texto", ""), grupo.get("titulo", ""),
                               cfg.get("brand", ""), cfg.get("aliases") or [],
                               voceros=cfg.get("voceros") or [])
    except Exception:
        return ""


def estado_grupo(grupo: dict, cfg: dict, tax: Optional[dict] = None,
                 subtemas_usados: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """El `state` que ve el modelo. Separa la evidencia de marca del contexto del hecho."""
    rubrica = cfg.get("criterio_texto") or cfg.get("criterio") or ""
    estado: Dict[str, Any] = {
        "entidad": cfg.get("brand") or "",
        "alias_y_formas_de_nombrarla": list(cfg.get("aliases") or []),
        "voceros": list(cfg.get("voceros") or []),
        "criterio_tono_del_cliente": rubrica,
        "lo_que_se_dice_de_la_entidad": _contexto_del_grupo(grupo, cfg),
        "contexto_del_hecho": {
            "titular": str(grupo.get("titulo") or "")[:220],
            "otros_titulares": [str(t)[:140] for t in (grupo.get("titulos_alt") or [])][:3],
            "cuerpo": str(grupo.get("texto") or "")[:TEXTO_COMPLETO_MAX],
            "menciones": int(grupo.get("n") or 1),
        },
    }
    if tax and tax.get("temas"):
        estado["cubos_disponibles"] = list(tax["temas"])
        estado["cubos_prohibidos"] = ["Otros", "Varios", "General", "Informacion general"]
    if subtemas_usados:
        estado["subtemas_ya_usados"] = [str(s) for s in subtemas_usados][-120:]
    return estado


def preguntas_grupo(tax: Optional[dict] = None,
                    candidatos_subtema: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Las cuatro preguntas independientes de un grupo (se evaluan en paralelo, mismo state)."""
    preguntas: Dict[str, Any] = {
        "mencion_entidad": pregunta_noul(
            "El texto nombra a la entidad objetivo, a alguno de sus alias o a uno de sus voceros",
            si="Aparece la entidad, un alias o un vocero, con participacion en el hecho",
            no="No aparece ninguno: el hecho es de un tercero o de otra entidad del mismo territorio"),
        "critica_dirigida": pregunta_noul(
            "El texto contiene una critica, denuncia, sancion, reclamo o exigencia DIRIGIDA a la entidad "
            "o a sus funcionarios (no a un tercero como el Gobierno Nacional, la Fiscalia o un contratista)",
            si="Hay señalamiento con blanco identificable: la entidad, un alias o un funcionario suyo",
            no="Solo hay hechos graves (robo, inundacion, accidente, cifra negativa) sin señalamiento, "
               "o la critica va dirigida a otro actor"),
        "solo_hecho_grave": pregunta_noul(
            "El texto describe un hecho que la entidad no causa ni controla (crimen, desastre natural, "
            "accidente, cifra social negativa) sin atribuirle responsabilidad",
            si="El hecho es externo y no se le atribuye a la entidad",
            no="La entidad es autora, responsable o queda bien o mal por su propia actuacion"),
        "tono": pregunta_choice(
            "¿Cual es el tono para la entidad objetivo segun el criterio del cliente y la evidencia que "
            "la menciona?",
            {
                "Positivo": "La entidad o su vocero es sujeto de un hecho favorable: obra entregada, avance, "
                            "beneficio, programa, reconocimiento, buena cifra, o alguien de la entidad habla.",
                "Neutro": "Informacion institucional sin juicio, agenda, tramites, contexto, cobertura de otra "
                          "entidad, o hecho grave sin señalamiento dirigido.",
                "Negativo": "Critica, denuncia, sancion, demanda, incumplimiento o evaluacion negativa dirigida "
                            "a la entidad, a su administracion o a sus funcionarios.",
            }),
    }
    if tax and tax.get("temas"):
        opciones = {str(t): None for t in tax["temas"]}
        preguntas["tema"] = pregunta_choice(
            "¿Cual de los cubos tematicos del cliente describe el hecho relacionado con la entidad? "
            "Elige exactamente uno de la lista.",
            opciones)
        preguntas["ningun_cubo_sirve"] = pregunta_noul(
            "Ninguno de los cubos de la lista describe el hecho con precision (hace falta uno nuevo especifico)",
            si="Todos los cubos son de otro asunto",
            no="Algun cubo de la lista si describe el hecho")
    if candidatos_subtema:
        preguntas["subtema"] = pregunta_choice(
            "¿Cual de los sub-temas ya usados corresponde al MISMO hecho? Elige uno, o null si ninguno "
            "corresponde (entonces se redacta uno nuevo).",
            dict({str(c): None for c in candidatos_subtema}, **{"__nuevo__": "No hay uno igual: hace falta un sub-tema nuevo"}))
    return preguntas


def construir_peticion(cfg: dict, grupos: Sequence[dict], tax: Optional[dict] = None,
                       subtemas_usados: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Un payload por grupo. Esto es lo que viaja tal cual a POST /v1/systemone."""
    peticiones = []
    for g in grupos:
        peticiones.append({
            "state": estado_grupo(g, cfg, tax, subtemas_usados),
            "model": MODELO_DEFECTO,
            "questions": preguntas_grupo(tax, candidatos_subtema=subtemas_usados),
        })
    return peticiones


# ============================================================================
# 4. Transporte (mismo espiritu que analyzer_tono_tema.llamar_llm)
# ============================================================================
def evaluar(peticion: Dict[str, Any], key: Optional[str] = None, timeout: int = 120,
            intentos: int = 3) -> Dict[str, Any]:
    """POST al endpoint de System One. Los question ids no salen del sobre (no llegan al modelo)."""
    credencial = api_key(key)
    if not credencial:
        raise TypeSafeSinCredencial(
            "Falta %s. Configurala en .streamlit/secrets.toml o como variable de entorno. "
            "No se cae en silencio a heuristicas (SPEC §0)." % VARIABLE_ENTORNO)
    cabeceras = {"Authorization": "Bearer %s" % credencial, "Content-Type": "application/json"}
    ultimo = ""
    for k in range(max(1, int(intentos))):
        try:
            r = requests.post(ENDPOINT, headers=cabeceras, json=peticion, timeout=timeout)
        except requests.RequestException as e:
            ultimo = str(e)[:200]
            time.sleep(2 + 3 * k)
            continue
        if r.status_code in (429, 500, 502, 503, 529):   # 529 = Overloaded (docs)
            ultimo = "HTTP %s" % r.status_code
            time.sleep(2 + 3 * k)
            continue
        if r.status_code == 401:
            raise TypeSafeError("HTTP 401: API key ausente o invalida (revisa Authorization).")
        if r.status_code != 200:
            raise TypeSafeError("HTTP %s: %s" % (r.status_code, r.text[:300]))
        return r.json()
    raise TypeSafeError("Fallo la llamada a TypeSafe: %s" % ultimo)


# ============================================================================
# 5. Composicion en codigo: TypeSafe propone, las reglas del SPEC deciden
# ============================================================================
def _answer(respuesta: dict, qid: str) -> Optional[dict]:
    return ((respuesta or {}).get("answers") or {}).get(qid)


def _noul(respuesta: dict, qid: str) -> Optional[float]:
    a = _answer(respuesta, qid)
    return float(a["noul"]) if a and isinstance(a.get("noul"), (int, float)) else None


def _choice(respuesta: dict, qid: str) -> Optional[dict]:
    a = _answer(respuesta, qid)
    return a if a and a.get("choice") else None


def componer(respuesta: dict, tax: Optional[dict] = None) -> Dict[str, Any]:
    """Convierte las respuestas tipadas en las etiquetas del motor, aplicando las guardas.

    Se ignoran las preguntas cuya respuesta no se usa (ramas no consumidas).
    """
    tono_choice = _choice(respuesta, "tono")
    tono = str(tono_choice.get("choice")) if tono_choice else ""
    if tono not in ("Positivo", "Neutro", "Negativo"):
        tono = "Neutro"
    mencion = _noul(respuesta, "mencion_entidad")
    critica = _noul(respuesta, "critica_dirigida")
    grave = _noul(respuesta, "solo_hecho_grave")
    tema_choice = _choice(respuesta, "tema")
    tema = str(tema_choice.get("choice")) if tema_choice else ""
    subtema_choice = _choice(respuesta, "subtema")
    subtema = str(subtema_choice.get("choice")) if subtema_choice else ""

    motivos: List[str] = []
    # SPEC §1 regla 3: sin mencion no hay tono con signo
    if mencion is not None and mencion < UMBRAL_MENCION and tono != "Neutro":
        motivos.append("sin_mencion(%.2f)" % mencion)
        tono = "Neutro"
    # SPEC §2.1: Negativo exige señalamiento dirigido
    if tono == "Negativo" and ((critica is not None and critica < UMBRAL_CRITICA) or
                               (grave is not None and grave >= UMBRAL_CRITICA)):
        motivos.append("negativo_sin_señalamiento(critica=%s,hecho_grave=%s)"
                       % (None if critica is None else round(critica, 2),
                          None if grave is None else round(grave, 2)))
        tono = "Neutro"

    # Tema: solo la lista cerrada del cliente. Fuera de la lista = rechazado, nunca inventado.
    tema_valido = ""
    if tema and tax and tax.get("temas"):
        exacto = next((t for t in tax["temas"] if str(t).strip().lower() == tema.strip().lower()), None)
        tema_valido = exacto or ""
        if not tema_valido:
            motivos.append("tema_fuera_de_la_lista(%s)" % tema[:40])
    elif tema:
        tema_valido = tema

    return {
        "tono": tono,
        "tema": tema_valido,
        "subtema": "" if subtema == "__nuevo__" else subtema,
        "motivos": motivos,
        "probabilidades": {
            "tono": (tono_choice or {}).get("probabilities"),
            "tema": (tema_choice or {}).get("probabilities"),
        },
        "confianza": {
            "tono": (tono_choice or {}).get("confidence"),
            "tema": (tema_choice or {}).get("confidence"),
        },
        "nouls": {"mencion_entidad": mencion, "critica_dirigida": critica, "solo_hecho_grave": grave},
    }


def verificar_grupos(cfg: dict, grupos: Sequence[dict], tax: Optional[dict] = None,
                     etiquetas_motor: Optional[Dict[int, dict]] = None,
                     key: Optional[str] = None, seco: bool = False,
                     subtemas_usados: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Corre la verificacion grupo por grupo y compara contra las etiquetas del motor.

    `seco=True` no llama a la API: devuelve el payload construido (asi se puede probar sin key).
    """
    subtemas = list(subtemas_usados or [])
    informe: List[Dict[str, Any]] = []
    for g in grupos:
        peticion = {
            "state": estado_grupo(g, cfg, tax, subtemas),
            "model": MODELO_DEFECTO,
            "questions": preguntas_grupo(tax, candidatos_subtema=subtemas),
        }
        if seco:
            informe.append({"grupo": g.get("grupo"), "titular": g.get("titulo", "")[:160],
                            "seco": True, "peticion": peticion})
            continue
        respuesta = evaluar(peticion, key=key)
        etiqueta = componer(respuesta, tax)
        motor = (etiquetas_motor or {}).get(g.get("grupo")) or {}
        difiere = {
            "tono": bool(motor.get("tono")) and str(motor["tono"]) != etiqueta["tono"],
            "tema": bool(motor.get("tema")) and str(motor["tema"]) != etiqueta["tema"],
            "subtema": bool(motor.get("subtema")) and str(motor["subtema"]) != etiqueta["subtema"],
        }
        informe.append({
            "grupo": g.get("grupo"),
            "titular": str(g.get("titulo", ""))[:160],
            "motor": {"tono": motor.get("tono"), "tema": motor.get("tema"), "subtema": motor.get("subtema")},
            "typesafe": etiqueta,
            "discrepancias": [k for k, v in difiere.items() if v],
            "usage": (respuesta or {}).get("usage") or {},
        })
    return informe


def _tono_de_choice(etiqueta: dict) -> str:
    """Normaliza el tono devuelto por TypeSafe al repertorio del motor."""
    t = str((etiqueta or {}).get("tono") or "").strip().capitalize()
    return t if t in ("Positivo", "Neutro", "Negativo") else "Neutro"


def asignar_tono_typesafe(cfg: dict, grupos: Sequence[dict],
                          key: Optional[str] = None, workers: int = 8,
                          umbral_confianza: float = 0.0,
                          progress: Optional[Callable] = None) -> Dict[int, dict]:
    """Etiqueta el TONO de cada grupo con TypeSafe (System One) y devuelve
    {grupo: {'tono', 'confianza', 'motivos'}}.

    Es la pieza que usa el motor como decisor de TONO cuando hay TYPESAFE_API_KEY:
    TypeSafe una eleccion tipada con probabilidad calibrada; el motor conserva
    Tema y Sub-tema. `umbral_confianza` descarta los grupos donde el modelo esta
    inseguro (NO se sobreescribe el tono del motor ahi).
    """
    if not grupos:
        return {}
    key = key or api_key()
    if not key:
        raise TypeSafeSinCredencial(
            "Falta %s para usar TypeSafe como clasificador de tono (SPEC §0). "
            "Configurala en .streamlit/secrets.toml o como variable de entorno." % VARIABLE_ENTORNO)
    subtemas = []            # el tono no depende de candidatos: se pasan vacios
    resultados: Dict[int, dict] = {}
    errores: List[str] = []

    def trabajo(g):
        peticion = {
            "state": estado_grupo(g, cfg, tax=None, subtemas_usados=None),
            "model": MODELO_DEFECTO,
            "questions": {"tono": preguntas_grupo(None)["tono"]},
        }
        try:
            respuesta = evaluar(peticion, key=key)
            etiqueta = componer(respuesta, tax=None)
            conf = (etiqueta.get("confianza") or {}).get("tono")
            return g["grupo"], {
                "tono": _tono_de_choice(etiqueta),
                "confianza": float(conf or 0.0),
                "motivos": etiqueta.get("motivos") or [],
            }, None
        except Exception as e:
            return g["grupo"], None, str(e)[:200]

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futuros = {ex.submit(trabajo, g): g for g in grupos}
        for fut in concurrent.futures.as_completed(futuros):
            gid, r, err = fut.result()
            if err:
                errores.append("g%s: %s" % (gid, err))
                continue
            if r and (umbral_confianza <= 0.0 or r["confianza"] >= umbral_confianza):
                resultados[gid] = r
            elif r:
                resultados[gid] = {"tono": _tono_de_choice(r), "confianza": r["confianza"],
                                   "motivos": list(r.get("motivos") or []) + ["baja_confianza"]}
    if progress:
        progress(90, "TypeSafe: tono asignado para %d grupos" % len(resultados))
    return resultados


def resumen(informe: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Conteos del informe. `prioridad` ordena los grupos donde hay desacuerdo de tono y el modelo
    esta muy seguro: son los candidatos mas probables a error del motor."""
    filas = [f for f in informe if not f.get("seco")]
    disc_tono = [f for f in filas if "tono" in (f.get("discrepancias") or [])]
    conf = [(f, (f["typesafe"].get("confianza") or {}).get("tono") or 0.0) for f in disc_tono]
    conf.sort(key=lambda x: -x[1])
    return {
        "grupos": len(filas),
        "discrepancias_tono": len(disc_tono),
        "discrepancias_tema": len([f for f in filas if "tema" in (f.get("discrepancias") or [])]),
        "negativos_bajados_a_neutro": len([f for f in filas
                                          if any("negativo_sin" in m for m in f["typesafe"]["motivos"])]),
        "sin_mencion": len([f for f in filas
                            if any("sin_mencion" in m for m in f["typesafe"]["motivos"])]),
        "tokens": {
            "entrada": sum(int((f.get("usage") or {}).get("input_tokens") or 0) for f in filas),
            "salida": sum(int((f.get("usage") or {}).get("output_tokens") or 0) for f in filas),
        },
        "prioridad_revision": [{"grupo": f.get("grupo"), "titular": f.get("titular"),
                               "motor": f["motor"]["tono"], "typesafe": f["typesafe"]["tono"],
                               "confianza": round(c, 3)} for f, c in conf[:15]],
    }


# ============================================================================
# 6. Revision del CODIGO: TypeSafe verifica reglas del SPEC en un modulo
# ============================================================================
# Ojo: no es un revisor de codigo en prosa (el modelo no genera texto). Son preguntas
# tipadas sobre el archivo como estado. Para reglas que se pueden comprobar por lectura
# estatica (grep/AST) el chequeo determinista es mejor; esto sirve para lo que exige
# juicio sobre el conjunto del modulo.
REGLAS_SPEC = {
    "guarda_positiva_simetrica": ("SPEC §2.2: el modulo aplica una guarda POSITIVA simetrica que sube "
                                  "Neutro a Positivo cuando la entidad es autora de la accion del titular"),
    "separa_bloques_en_prompt": ("SPEC §1: el prompt de etiquetado separa LO QUE SE DICE DE LA ENTIDAD "
                                 "(decide el tono) del CONTEXTO DEL HECHO (no decide el tono)"),
    "sin_fallback_silencioso": ("SPEC §0: si falta la API key el modulo avisa y se detiene; no cae en "
                                "silencio a heuristicas"),
    "tono_por_grupo_no_por_fila": ("SPEC §1 regla 4: el tono se decide por GRUPO de notas equivalentes "
                                   "y nunca fila por fila"),
}


def preguntas_especificacion(codigo: str) -> Dict[str, Any]:
    """Preguntas tipadas sobre el codigo como estado (patron 'Verification' de la doc)."""
    return {
        "guarda_positiva_simetrica": pregunta_noul(
            REGLAS_SPEC["guarda_positiva_simetrica"] + "¿Esta implementada en el codigo?",
            si="Hay una funcion o rama que sube Neutro a Positivo por autoria de la accion",
            no="Solo existe la guarda que baja Negativo a Neutro"),
        "separa_bloques_en_prompt": pregunta_noul(
            REGLAS_SPEC["separa_bloques_en_prompt"] + "¿Esta implementada en el codigo?",
            si="El prompt marca explicitamente cual bloque decide el tono",
            no="El prompt pasa el titular y el cuerpo juntos, sin separar la evidencia de marca"),
        "sin_fallback_silencioso": pregunta_noul(
            REGLAS_SPEC["sin_fallback_silencioso"] + "¿Se cumple?",
            si="Hay un error explicito cuando falta la credencial",
            no="El codigo sigue con heuristicas sin avisar"),
        "tono_por_grupo_no_por_fila": pregunta_noul(
            REGLAS_SPEC["tono_por_grupo_no_por_fila"] + "¿Se cumple?",
            si="Las etiquetas se calculan una vez por grupo y se propagan a sus filas",
            no="Cada fila se etiqueta por separado"),
    }


def revisar_codigo(codigo: str, key: Optional[str] = None, timeout: int = 120) -> Dict[str, Any]:
    """Pregunta por las reglas obligatorias del SPEC sobre el contenido de un modulo."""
    peticion = {
        "state": {"archivo": "analyzer_tono_tema.py", "codigo": codigo},
        "model": MODELO_DEFECTO,
        "questions": preguntas_especificacion(codigo),
    }
    respuesta = evaluar(peticion, key=key, timeout=timeout)
    salida = {}
    for qid in preguntas_especificacion(codigo):
        v = _noul(respuesta, qid)
        salida[qid] = {"cumple": None if v is None else v >= 0.5, "prob": v,
                       "regla": REGLAS_SPEC[qid]}
    return salida


# ============================================================================
# 7. CLI: informe sobre un dossier real (necesita la key)
# ============================================================================
def _cargar_grupos(xlsx: str, marca: str, alias: Sequence[str], voceros: Sequence[str],
                   tax: Optional[dict], limite: Optional[int]) -> List[dict]:
    import pandas as pd
    from analyzer_tono_tema import _contexto_marca, construir_grupos
    km = {"titulo": "Título", "resumen": "Resumen - Aclaracion"}
    try:
        from pipeline import KEY_MAP
        km = KEY_MAP
    except Exception:
        pass
    df = pd.read_excel(xlsx, dtype=str).fillna("")
    rows = df.to_dict("records")
    for r in rows:
        r.setdefault("is_duplicate", False)
        r["Contexto analizado"] = _contexto_marca(
            str(r.get("Resumen - Aclaracion") or r.get("resumen corto") or ""),
            str(r.get(km.get("titulo", "Título")) or ""), marca, list(alias), voceros=list(voceros))
    grupos, _mapa = construir_grupos(rows, km)
    for g in grupos:
        g["contexto"] = str(rows[g["idxs"][0]].get("Contexto analizado") or "")
        g["motor"] = {"tono": str(rows[g["idxs"][0]].get("Tono_IA") or ""),
                      "tema": str(rows[g["idxs"][0]].get("Tema_IA") or ""),
                      "subtema": str(rows[g["idxs"][0]].get("Subtema_IA") or "")}
    if limite:
        grupos = grupos[:int(limite)]
    return grupos


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verifica las etiquetas del motor con TypeSafe (System One).")
    ap.add_argument("--xlsx", required=True, help="dossier exportado por el motor")
    ap.add_argument("--marca", required=True)
    ap.add_argument("--alias", default="", help="separados por coma o punto y coma")
    ap.add_argument("--voceros", default="", help="separados por coma o punto y coma")
    ap.add_argument("--criterio", default="", help="texto de la rubrica de tono del cliente")
    ap.add_argument("--taxonomia", default="", help="nombre de taxonomia o ruta a un JSON con 'temas'")
    ap.add_argument("--limite", type=int, default=0, help="primeros N grupos (0 = todos)")
    ap.add_argument("--seco", action="store_true", help="no llama a la API: imprime el payload")
    ap.add_argument("--salida", default="", help="ruta del JSON del informe")
    args = ap.parse_args(argv)

    import re as _re
    alias = [a.strip() for a in _re.split(r"[,;\n]", args.alias) if a.strip()]
    voceros = [v.strip() for v in _re.split(r"[,;\n]", args.voceros) if v.strip()]
    tax = None
    if args.taxonomia:
        if os.path.exists(args.taxonomia):
            with open(args.taxonomia, encoding="utf-8") as fh:
                tax = json.load(fh)
        else:
            from catalogo_tono_tema import taxonomia_por_nombre
            tax = taxonomia_por_nombre(args.taxonomia)

    grupos = _cargar_grupos(args.xlsx, args.marca, alias, voceros, tax, args.limite)
    cfg = {"brand": args.marca, "aliases": alias, "voceros": voceros, "criterio_texto": args.criterio}
    if args.seco:
        for p in construir_peticion(cfg, grupos, tax)[:1]:
            print(json.dumps(p, ensure_ascii=False, indent=2))
        print("-- seco: %d grupos, 0 llamadas a la API" % len(grupos))
        return 0

    motor = {g["grupo"]: g.get("motor") or {} for g in grupos}
    informe = verificar_grupos(cfg, grupos, tax, etiquetas_motor=motor,
                               subtemas_usados=[(g.get("motor") or {}).get("subtema") for g in grupos])
    salida = {"resumen": resumen(informe), "detalle": informe}
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as fh:
            json.dump(salida, fh, ensure_ascii=False, indent=2)
        print("Informe: %s" % args.salida)
    print(json.dumps(salida["resumen"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
