from enum import Enum

from tree_sitter import Tree

import os
from pathlib import Path
from typing import Optional

from tree_sitter import Language, Parser, Node

class Lang(Enum):
    JAVA = ".java"
    XML = ".xml"
    PYTHON = ".py"
    C = ".c"
    PHP = ".php"

project_path = Path(*Path(__file__).parts[:-3])
Language.build_library(
    os.path.join(project_path, "utils/parserTool", "my-languages.so"),

    [
        os.path.join(project_path, "utils/parserTool", "tree-sitter-c"),
        os.path.join(project_path, "utils/parserTool", "tree-sitter-java"),
        os.path.join(project_path, "utils/parserTool", "tree-sitter-python"),
        os.path.join(project_path, "utils/parserTool", "tree-sitter-php"),
    ]
)

def java():
    parser = Parser()
    parser.set_language(Language(os.path.join(
        project_path, "utils/parserTool", "my-languages.so"), "java"))
    return parser

def python():
    parser = Parser()
    parser.set_language(Language(os.path.join(
        project_path, "utils/parserTool", "my-languages.so"), "python"))
    return parser

def c():
    parser = Parser()
    parser.set_language(Language(os.path.join(
        project_path, "utils/parserTool", "my-languages.so"), "c"))
    return parser

def php():
    parser = Parser()
    parser.set_language(Language(os.path.join(
        project_path, "utils/parserTool", "my-languages.so"), "php"))
    return parser

def tree_sitter_ast(source_code: str,lang: Lang) -> Tree:
    """
    Parse the source code in a specified format into a Tree Sitter AST.
    :param source_code: string with the source code in it.
    :param lang: the source code Lang.
    :return: Tree Sitter AST.
    """
    if lang == Lang.JAVA:
        return java().parse(bytes(source_code, "utf8"))
    elif lang == Lang.PYTHON:
        return python().parse(bytes(source_code, "utf8"))
    elif lang == Lang.C:
        return c().parse(bytes(source_code, "utf8"))
    elif lang == Lang.PHP:
        return php().parse(bytes(source_code, "utf8"))
    else:
        raise NotImplementedError()
