"""
Market Bridge — AI-Powered Investment Research Synthesis
========================================================
Top-down investment platform integrating macroeconomic regime context,
SEC filing analysis, and LLM-powered research synthesis.

Architecture:
  Data Sources (sec-api.io, FRED, Polygon)
       ↓
  Chunking (Semantic for 8-K, Template for 10-K/10-Q)
       ↓
  Synthesis (Anthropic Claude API)
       ↓
  Structured Investment Intelligence (JSON)
"""

__version__ = "1.0.0"
