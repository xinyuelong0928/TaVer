from tree_sitter import Language, Parser

Language.build_library(
  'my-languages.so',
  
  [
    'tree-sitter-java',
    'tree-sitter-c',
    'tree-sitter-python',
    'tree-sitter-php',
  ]
)

