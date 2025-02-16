uv build
uvx twine check dist/*
uvx twine upload --skip-existing dist/*
pause
