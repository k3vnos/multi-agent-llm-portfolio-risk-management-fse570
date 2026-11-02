"""OPTIONAL explanation layer (M6): grounded multi-agent explanations of saved alerts.

Reads saved run artifacts only and never writes back into predictions or metrics. The core package `riskcausal`
cannot import this package (enforced by a test). LLM inference is local (Ollama on 127.0.0.1) or disabled.
"""
