project = 'reafferent.net'
copyright = '2024, Kayce Basques'
author = 'Kayce Basques'
release = '0.0.0'
extensions = ['sphinx-pigweed']
exclude_patterns = [
    '.github',
    '.gitignore',
    'Makefile',
    'README.md',
    '_build',
    'boostrap.sh',
    'requirements.txt',
    'venv'
]
pygments_style = 'sphinx'
html_theme = 'sphinx-pigweed'
html_static_path = ['_static']
html_permalinks_icon = '#'
pw_banner_text = 'Hello, world!'