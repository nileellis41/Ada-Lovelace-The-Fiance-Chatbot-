"""
Market Bridge — Core Pipeline Orchestrator
============================================
Orchestrates the full analysis pipeline:
  SEC-API.io → Chunking (semantic/template) → Claude Synthesis

Three primary workflows:
  1. Earnings Analysis:  8-K → semantic chunking → Claude
  2. Annual Analysis:    10-K → template chunking → Claude
  3. Quarterly Analysis: 10-Q → template chunking → Claude

Each workflow enriches the prompt with:
  - FRED macro context (yield curve, VIX, credit spreads)
  - Polygon market context (price action, market cap)
  - XBRL structured financials (when available)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from market_bridge.data_sources import SecApiClient, SecFiling, FredClient, PolygonClient
from market_bridge.chunking import SemanticChunker, TemplateChunker, Chunk
from market_bridge.synthesis import SynthesisEngine, AnalysisResult
from market_bridge.config.settings import AppConfig, ChunkingConfig
from market_bridge.utils.helpers import get_logger

logger = get_logger("pipeline")


@dataclass
class PipelineResult:
    """Complete output from a pipeline run."""
    ticker: str
    analysis: AnalysisResult
    filing: Optional[SecFiling] = None
    chunks: List[Chunk] = None
    macro_context: str = ""
    market_context: str = ""
    xbrl_context: str = ""
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "analysis_type": self.analysis.analysis_type,
            "signal": self.analysis.signal,
            "conviction": self.analysis.conviction,
            "summary": self.analysis.summary,
            "structured_data": self.analysis.structured_data,
            "filing_date": self.filing.filed_at if self.filing else "",
            "period": self.filing.period_of_report if self.filing else "",
            "chunks_used": self.analysis.chunks_used,
            "tokens": {
                "input": self.analysis.total_input_tokens,
                "output": self.analysis.total_output_tokens,
            },
            "model": self.analysis.model,
        }


class MarketBridgePipeline:
    """
    Main orchestrator for Market Bridge analysis workflows.
    
    Usage:
        pipeline = MarketBridgePipeline()
        result = pipeline.analyze_earnings("AAPL")
        result = pipeline.analyze_annual("MSFT")
        result = pipeline.analyze_quarterly("GOOGL")
        result = pipeline.custom_query("NVDA", "What are the AI-related revenue drivers?")
    """
    
    def __init__(
        self,
        sec_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        polygon_api_key: Optional[str] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or AppConfig()
        
        # Data sources
        self.sec = SecApiClient(api_key=sec_api_key)
        self.fred = FredClient(api_key=fred_api_key)
        self.polygon = PolygonClient(api_key=polygon_api_key)
        
        # Chunkers
        self.semantic_chunker = SemanticChunker(config=self.config.chunking)
        self.template_chunker = TemplateChunker(config=self.config.chunking)
        
        # Synthesis
        self.synthesis = SynthesisEngine(api_key=anthropic_api_key)
    
    def _build_context(self, ticker: str, xbrl_data: Optional[Dict] = None) -> tuple:
        """Build macro + market + XBRL context strings."""
        macro_ctx = ""
        market_ctx = ""
        xbrl_ctx = ""
        
        try:
            macro_ctx = self.fred.format_macro_context()
        except Exception as e:
            logger.warning(f"Failed to build macro context: {e}")
        
        try:
            market_ctx = self.polygon.format_market_context(ticker)
        except Exception as e:
            logger.warning(f"Failed to build market context: {e}")
        
        if xbrl_data:
            try:
                xbrl_ctx = self.sec.format_xbrl_context(xbrl_data)
            except Exception as e:
                logger.warning(f"Failed to format XBRL context: {e}")
        
        combined = "\n\n".join(filter(None, [macro_ctx, market_ctx, xbrl_ctx]))
        return combined, macro_ctx, market_ctx, xbrl_ctx
    
    # ═══════════════════════════════════════════
    # EARNINGS ANALYSIS (8-K → Semantic Chunking)
    # ═══════════════════════════════════════════
    def analyze_earnings(self, ticker: str) -> PipelineResult:
        """
        Full earnings analysis pipeline:
          1. Pull latest earnings 8-K via sec-api.io
          2. Semantic chunking on transcript text
          3. Build macro/market context
          4. Claude synthesis → structured earnings analysis
        """
        logger.info(f"═══ EARNINGS ANALYSIS: {ticker} ═══")
        
        # Step 1: Pull earnings transcript
        filing = self.sec.pull_earnings_transcript(ticker)
        if not filing:
            raise ValueError(f"No earnings filing found for {ticker}")
        
        # Step 2: Semantic chunking
        # Combine all extracted sections into one text block for semantic chunking
        full_text = "\n\n".join(filing.sections.values())
        chunks = self.semantic_chunker.chunk(
            text=full_text,
            source=f"{ticker} 8-K ({filing.filed_at})",
            filing_ticker=ticker,
        )
        
        # Step 3: Context
        combined_ctx, macro_ctx, market_ctx, xbrl_ctx = self._build_context(ticker)
        
        # Step 4: Synthesis
        analysis = self.synthesis.analyze_earnings(
            chunks=chunks,
            ticker=ticker,
            company_name=filing.company_name,
            macro_context=combined_ctx,
        )
        
        logger.info(f"═══ EARNINGS RESULT: {ticker} → {analysis.signal} ({analysis.conviction}) ═══")
        
        return PipelineResult(
            ticker=ticker,
            analysis=analysis,
            filing=filing,
            chunks=chunks,
            macro_context=macro_ctx,
            market_context=market_ctx,
        )
    
    # ═══════════════════════════════════════════
    # ANNUAL ANALYSIS (10-K → Template Chunking)
    # ═══════════════════════════════════════════
    def analyze_annual(self, ticker: str, priority_only: bool = True) -> PipelineResult:
        """
        Full annual report analysis pipeline:
          1. Pull latest 10-K via sec-api.io + extract sections
          2. Template chunking on extracted sections
          3. Optional XBRL financials
          4. Build macro/market context
          5. Claude synthesis → structured filing analysis
        """
        logger.info(f"═══ ANNUAL ANALYSIS: {ticker} ═══")
        
        # Step 1: Pull and extract 10-K
        filing = self.sec.pull_annual_report(
            ticker, with_xbrl=True, priority_only=priority_only
        )
        if not filing:
            raise ValueError(f"No 10-K filing found for {ticker}")
        
        # Step 2: Template chunking
        chunks = self.template_chunker.chunk_with_context_window(
            sections=filing.sections,
            max_total_tokens=80000,
            source=f"{ticker} 10-K ({filing.filed_at})",
            filing_ticker=ticker,
        )
        
        # Step 3: Context (including XBRL if available)
        xbrl = filing.metadata.get("xbrl_financials")
        combined_ctx, macro_ctx, market_ctx, xbrl_ctx = self._build_context(ticker, xbrl)
        
        # Step 4: Synthesis
        analysis = self.synthesis.analyze_filing(
            chunks=chunks,
            ticker=ticker,
            company_name=filing.company_name,
            filing_type="10-K",
            macro_context=combined_ctx,
        )
        
        logger.info(f"═══ ANNUAL RESULT: {ticker} → {analysis.signal} ({analysis.conviction}) ═══")
        
        return PipelineResult(
            ticker=ticker,
            analysis=analysis,
            filing=filing,
            chunks=chunks,
            macro_context=macro_ctx,
            market_context=market_ctx,
            xbrl_context=xbrl_ctx,
        )
    
    # ═══════════════════════════════════════════
    # QUARTERLY ANALYSIS (10-Q → Template Chunking)
    # ═══════════════════════════════════════════
    def analyze_quarterly(self, ticker: str, priority_only: bool = True) -> PipelineResult:
        """
        Full quarterly report analysis pipeline:
          1. Pull latest 10-Q via sec-api.io + extract sections
          2. Template chunking on extracted sections
          3. Optional XBRL financials
          4. Build macro/market context
          5. Claude synthesis → structured filing analysis
        """
        logger.info(f"═══ QUARTERLY ANALYSIS: {ticker} ═══")
        
        filing = self.sec.pull_quarterly_report(
            ticker, with_xbrl=True, priority_only=priority_only
        )
        if not filing:
            raise ValueError(f"No 10-Q filing found for {ticker}")
        
        chunks = self.template_chunker.chunk_with_context_window(
            sections=filing.sections,
            max_total_tokens=80000,
            source=f"{ticker} 10-Q ({filing.filed_at})",
            filing_ticker=ticker,
        )
        
        xbrl = filing.metadata.get("xbrl_financials")
        combined_ctx, macro_ctx, market_ctx, xbrl_ctx = self._build_context(ticker, xbrl)
        
        analysis = self.synthesis.analyze_filing(
            chunks=chunks,
            ticker=ticker,
            company_name=filing.company_name,
            filing_type="10-Q",
            macro_context=combined_ctx,
        )
        
        logger.info(f"═══ QUARTERLY RESULT: {ticker} → {analysis.signal} ({analysis.conviction}) ═══")
        
        return PipelineResult(
            ticker=ticker,
            analysis=analysis,
            filing=filing,
            chunks=chunks,
            macro_context=macro_ctx,
            market_context=market_ctx,
            xbrl_context=xbrl_ctx,
        )
    
    # ═══════════════════════════════════════════
    # CUSTOM QUERY
    # ═══════════════════════════════════════════
    def custom_query(
        self,
        ticker: str,
        question: str,
        filing_type: str = "10-K",
    ) -> PipelineResult:
        """
        Run an ad-hoc analytical question against a filing.
        
        Examples:
            pipeline.custom_query("NVDA", "What are the AI-related revenue drivers?")
            pipeline.custom_query("JPM", "What is the credit loss provision trajectory?", "10-Q")
        """
        logger.info(f"═══ CUSTOM QUERY: {ticker} — {question} ═══")
        
        # Pull the right filing type
        if filing_type == "8-K":
            filing = self.sec.pull_earnings_transcript(ticker)
            if filing:
                full_text = "\n\n".join(filing.sections.values())
                chunks = self.semantic_chunker.chunk(full_text, source=f"{ticker} 8-K", filing_ticker=ticker)
            else:
                raise ValueError(f"No 8-K found for {ticker}")
        else:
            if filing_type == "10-K":
                filing = self.sec.pull_annual_report(ticker, with_xbrl=False)
            else:
                filing = self.sec.pull_quarterly_report(ticker, with_xbrl=False)
            
            if not filing:
                raise ValueError(f"No {filing_type} found for {ticker}")
            
            chunks = self.template_chunker.chunk(
                sections=filing.sections,
                source=f"{ticker} {filing_type}",
                filing_ticker=ticker,
            )
        
        combined_ctx, macro_ctx, market_ctx, _ = self._build_context(ticker)
        
        analysis = self.synthesis.custom_analysis(
            chunks=chunks,
            question=question,
            ticker=ticker,
            macro_context=combined_ctx,
        )
        
        return PipelineResult(
            ticker=ticker,
            analysis=analysis,
            filing=filing,
            chunks=chunks,
            macro_context=macro_ctx,
            market_context=market_ctx,
        )
