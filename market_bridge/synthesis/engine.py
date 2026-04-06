"""
Market Bridge — LLM Synthesis Engine
=====================================
Sends chunked SEC filing content + macroeconomic context to Claude API
for structured investment intelligence synthesis.

Pipeline:
  1. Receive chunks (semantic or template) + macro context
  2. Assemble prompt from templates with chunk text + context
  3. Call Anthropic Claude API
  4. Parse structured JSON response
  5. Return typed analysis objects
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import anthropic

from market_bridge.config.settings import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_MAX_TOKENS,
    SYNTHESIS_SYSTEM_PROMPT,
    EARNINGS_ANALYSIS_PROMPT,
    FILING_ANALYSIS_PROMPT,
)
from market_bridge.chunking.semantic_chunker import Chunk
from market_bridge.utils.helpers import get_logger

logger = get_logger("synthesis")


@dataclass
class AnalysisResult:
    """Structured output from LLM synthesis."""
    ticker: str
    analysis_type: str          # "earnings" | "annual" | "quarterly"
    signal: str                 # "bullish" | "bearish" | "neutral"
    conviction: str             # "high" | "medium" | "low"
    summary: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    chunks_used: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    model: str = ""
    raw_response: str = ""


class SynthesisEngine:
    """
    LLM synthesis engine using Anthropic Claude API.
    
    Assembles prompts from chunked filing content and macro context,
    calls Claude, and parses structured JSON responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or ANTHROPIC_MODEL
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — synthesis will fail.")
    
    def _format_chunks_for_prompt(self, chunks: List[Chunk], max_chars: int = 100000) -> str:
        """Format chunks into a single text block for the prompt."""
        lines = []
        total_chars = 0
        
        for chunk in chunks:
            header = f"\n--- [{chunk.section_label}] (chunk {chunk.chunk_id}, {chunk.chunk_type}) ---\n"
            
            if total_chars + len(chunk.text) + len(header) > max_chars:
                lines.append("\n[... additional content truncated for context window ...]")
                break
            
            lines.append(header)
            lines.append(chunk.text)
            total_chars += len(chunk.text) + len(header)
        
        return "\n".join(lines)
    
    def _parse_json_response(self, text: str) -> Dict:
        """Extract and parse JSON from Claude's response."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON block
        import re
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning("Failed to parse JSON from Claude response — returning raw text")
        return {"raw_response": text, "parse_error": True}
    
    def _call_claude(
        self,
        user_prompt: str,
        system_prompt: str = SYNTHESIS_SYSTEM_PROMPT,
    ) -> tuple:
        """
        Make a single call to the Anthropic Claude API.
        
        Returns:
            (response_text, input_tokens, output_tokens)
        """
        if not self.client:
            raise RuntimeError("Anthropic client not initialized — check API key.")
        
        logger.info(f"Calling Claude ({self.model}) — prompt: {len(user_prompt)} chars")
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text
        
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        
        logger.info(f"Claude response: {len(response_text)} chars, "
                     f"tokens: {input_tokens} in / {output_tokens} out")
        
        return response_text, input_tokens, output_tokens
    
    # ═══════════════════════════════════════════
    # EARNINGS TRANSCRIPT ANALYSIS
    # ═══════════════════════════════════════════
    def analyze_earnings(
        self,
        chunks: List[Chunk],
        ticker: str,
        company_name: str,
        macro_context: str = "",
    ) -> AnalysisResult:
        """
        Synthesize earnings transcript chunks into structured analysis.
        
        Args:
            chunks: Semantically-chunked 8-K earnings transcript
            ticker: Stock ticker
            company_name: Full company name
            macro_context: Formatted macro/market context string
        
        Returns:
            AnalysisResult with earnings-specific structured data
        """
        chunks_text = self._format_chunks_for_prompt(chunks)
        
        prompt = EARNINGS_ANALYSIS_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            chunks=chunks_text,
            macro_context=macro_context or "N/A — macro context unavailable",
        )
        
        response_text, in_tokens, out_tokens = self._call_claude(prompt)
        parsed = self._parse_json_response(response_text)
        
        return AnalysisResult(
            ticker=ticker,
            analysis_type="earnings",
            signal=parsed.get("signal", "neutral"),
            conviction=parsed.get("conviction", "medium"),
            summary=parsed.get("summary", ""),
            structured_data=parsed,
            chunks_used=len(chunks),
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            model=self.model,
            raw_response=response_text,
        )
    
    # ═══════════════════════════════════════════
    # ANNUAL / QUARTERLY FILING ANALYSIS
    # ═══════════════════════════════════════════
    def analyze_filing(
        self,
        chunks: List[Chunk],
        ticker: str,
        company_name: str,
        filing_type: str = "10-K",
        macro_context: str = "",
    ) -> AnalysisResult:
        """
        Synthesize 10-K/10-Q filing chunks into structured analysis.
        
        Args:
            chunks: Template-chunked filing sections
            ticker: Stock ticker
            company_name: Full company name
            filing_type: "10-K" or "10-Q"
            macro_context: Formatted macro/market context string
        
        Returns:
            AnalysisResult with filing-specific structured data
        """
        sections_text = self._format_chunks_for_prompt(chunks)
        
        prompt = FILING_ANALYSIS_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            filing_type=filing_type,
            sections=sections_text,
            macro_context=macro_context or "N/A — macro context unavailable",
        )
        
        response_text, in_tokens, out_tokens = self._call_claude(prompt)
        parsed = self._parse_json_response(response_text)
        
        analysis_type = "annual" if "10-K" in filing_type else "quarterly"
        
        return AnalysisResult(
            ticker=ticker,
            analysis_type=analysis_type,
            signal=parsed.get("signal", "neutral"),
            conviction=parsed.get("conviction", "medium"),
            summary=parsed.get("summary", ""),
            structured_data=parsed,
            chunks_used=len(chunks),
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            model=self.model,
            raw_response=response_text,
        )
    
    # ═══════════════════════════════════════════
    # CUSTOM / AD-HOC ANALYSIS
    # ═══════════════════════════════════════════
    def custom_analysis(
        self,
        chunks: List[Chunk],
        question: str,
        ticker: str = "",
        macro_context: str = "",
    ) -> AnalysisResult:
        """
        Run a custom analytical question against filing chunks.
        Useful for ad-hoc queries like "What are the key litigation risks?"
        """
        chunks_text = self._format_chunks_for_prompt(chunks)
        
        prompt = f"""Analyze the following SEC filing content for {ticker}.

USER QUESTION: {question}

FILING CONTENT:
{chunks_text}

MACRO CONTEXT:
{macro_context or "N/A"}

Provide a thorough, quantitative analysis answering the user's question.
Cite specific data points from the filing. Format your response as JSON:
{{
  "ticker": "{ticker}",
  "question": "{question}",
  "answer": str,
  "key_data_points": [str],
  "signal": "bullish|bearish|neutral",
  "conviction": "high|medium|low",
  "summary": str
}}
"""
        
        response_text, in_tokens, out_tokens = self._call_claude(prompt)
        parsed = self._parse_json_response(response_text)
        
        return AnalysisResult(
            ticker=ticker,
            analysis_type="custom",
            signal=parsed.get("signal", "neutral"),
            conviction=parsed.get("conviction", "medium"),
            summary=parsed.get("summary", parsed.get("answer", "")),
            structured_data=parsed,
            chunks_used=len(chunks),
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            model=self.model,
            raw_response=response_text,
        )
