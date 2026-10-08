"""Behavior tests for the native Qwen ASR leading `<asr_text>` decode marker fix.

The helper under test (`TRTEdgeLLMASRBackend._strip_language_prefix`) is
extracted via AST from the ACTUAL canonical backend file and executed in a
controlled namespace (builtins/typing only) — no full-module import, no
numpy/GPU/device access. Provenance (backend path, file SHA-256, extracted
function SHA-256) is printed for evidence.
"""

import ast
import builtins
import hashlib
import importlib.util
import typing

# The voxedge that the server would import (installed wheel or PYTHONPATH).
BACKEND_PATH = importlib.util.find_spec(
    "voxedge.backends.jetson.trt_edge_llm_asr"
).origin
CLASS_NAME = "TRTEdgeLLMASRBackend"
HELPER_NAME = "_strip_language_prefix"


def _extract_helper():
    with open(BACKEND_PATH, "r", encoding="utf-8") as fh:
        source = fh.read()
    file_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    tree = ast.parse(source)
    helper_function = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME:
            for child in node.body:
                if (
                    isinstance(child, ast.FunctionDef)
                    and child.name == HELPER_NAME
                ):
                    helper_function = child
                    break
            break
    assert helper_function is not None, (
        f"{CLASS_NAME}.{HELPER_NAME} not found in AST of {BACKEND_PATH}"
    )
    decorator_list = helper_function.decorator_list
    assert len(decorator_list) == 1, "expected exactly the staticmethod decorator"
    assert isinstance(decorator_list[0], ast.Name)
    assert decorator_list[0].id == "staticmethod"
    namespace = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if not name.startswith("_")
        },
        "Optional": typing.Optional,
    }
    module_ast = ast.Module(body=[helper_function], type_ignores=[])
    exec(compile(module_ast, BACKEND_PATH, "exec"), namespace)
    helper = namespace[HELPER_NAME]
    function_source = ast.get_source_segment(source, helper_function) or ""
    function_sha256 = hashlib.sha256(
        function_source.encode("utf-8")
    ).hexdigest()
    return helper, file_sha256, function_sha256


HELPER, BACKEND_SHA256, HELPER_SHA256 = _extract_helper()


def _check(description, text, expected_text, expected_language):
    got_text, got_language = HELPER(text)
    assert got_text == expected_text, (
        f"{description}: text {got_text!r} != {expected_text!r}"
    )
    assert got_language == expected_language, (
        f"{description}: language {got_language!r} != {expected_language!r}"
    )
    print(f"PASS {description}: {text!r} -> ({expected_text!r}, {expected_language!r})")


def test_asr_native_text_marker():
    # 1. Actual NX HTTP smoke sample: leading marker only -> clean body.
    _check(
        "actual NX sample marker only",
        "<asr_text>Concord returned to its place amidst the tents.",
        "Concord returned to its place amidst the tents.",
        None,
    )
    # 2. Known-language header, unspaced before marker.
    _check(
        "English header unspaced marker",
        "language English<asr_text> hello world",
        "hello world",
        "English",
    )
    # 3. Known-language header, spaced before marker.
    _check(
        "English header spaced marker",
        "language English <asr_text> good morning",
        "good morning",
        "English",
    )
    # 4. Known-language header with unicode body, marker boundary.
    _check(
        "Chinese header unicode body",
        "language Chinese <asr_text>你好世界",
        "你好世界",
        "Chinese",
    )
    # 5. Unknown single-word label with hard marker boundary (no space
    #    between label and marker) — label must not swallow the marker.
    _check(
        "unknown single label marker boundary",
        "language Foo<asr_text> bar baz",
        "bar baz",
        "Foo",
    )
    # 6. Valid header with empty body: clean empty text, language detected.
    _check(
        "valid header empty body",
        "language English <asr_text>",
        "",
        "English",
    )
    # 7. Plain spoken body keeps an embedded literal marker untouched.
    _check(
        "embedded literal marker preserved",
        "I said <asr_text> out loud",
        "I said <asr_text> out loud",
        None,
    )
    # 8. Legacy known/unknown label behavior without marker unchanged.
    _check("legacy known label no marker", "language English hello", "hello", "English")
    _check("legacy unknown label no marker", "language Foo bar", "bar", "Foo")
    # 9. Incomplete leading '<asr_' token is NOT inventedly removed.
    _check(
        "incomplete leading token preserved",
        "<asr_text is not a marker yet",
        "<asr_text is not a marker yet",
        None,
    )
    # 10. Empty header label must not invent a language nor delete the body.
    _check(
        "empty header label preserves input",
        "language <asr_text> hello",
        "language <asr_text> hello",
        None,
    )
    # 10b. Whitespace-only (multi-space) header before the marker is still an
    #      empty header: preserve input, invent no language.
    _check(
        "two-space whitespace-only header preserves input",
        "language  <asr_text> hello",
        "language  <asr_text> hello",
        None,
    )
    # 10c. Tab-only whitespace header before the marker: same preservation.
    _check(
        "tab-only whitespace header preserves input",
        "language \t<asr_text> hello",
        "language \t<asr_text> hello",
        None,
    )
    # 10d. Single-token header with empty body: clean empty text, language
    #      detected (boundary label variant).
    _check(
        "single-token header empty body",
        "language Englishman <asr_text>",
        "",
        "Englishman",
    )
    # 10e. Single-word prefix boundary: 'Englishman' is the whole native label,
    #      not an 'English' prefix swallowing 'man<asr_text>'.
    _check(
        "single-word prefix boundary Englishman",
        "language Englishman<asr_text> hello there",
        "hello there",
        "Englishman",
    )

