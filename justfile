default: help

help:
    just --list

install:
    uv sync

run: install
    uv run python3 main.py