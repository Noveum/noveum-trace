"""
Document Processing Example with Noveum Trace SDK

This example demonstrates how to trace a document processing pipeline
that includes text extraction, summarization, and analysis.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import noveum_trace


@dataclass
class Document:
    """Represents a document to be processed."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Represents the result of document processing."""
    document_id: str
    summary: str
    key_points: List[str]
    sentiment: str
    topics: List[str]
    processing_time_ms: float


class DocumentProcessor:
    """A document processing pipeline with comprehensive tracing."""
    
    def __init__(self):
        # Initialize tracing
        noveum_trace.init(
            service_name="document-processor",
            environment="demo",
            log_directory="./document_traces",
            capture_content=True
        )
        
        self.tracer = noveum_trace.get_tracer()
    
    def process_document(self, document: Document) -> ProcessingResult:
        """Process a single document through the complete pipeline."""
        with self.tracer.start_span("document_processing") as main_span:
            main_span.set_attribute("document.id", document.id)
            main_span.set_attribute("document.title", document.title)
            main_span.set_attribute("document.content_length", len(document.content))
            main_span.set_attribute("document.word_count", len(document.content.split()))
            
            # Add document metadata
            for key, value in document.metadata.items():
                main_span.set_attribute(f"document.metadata.{key}", str(value))
            
            start_time = time.time()
            
            # Step 1: Text preprocessing
            processed_text = self._preprocess_text(document)
            
            # Step 2: Generate summary
            summary = self._generate_summary(document, processed_text)
            
            # Step 3: Extract key points
            key_points = self._extract_key_points(document, processed_text)
            
            # Step 4: Analyze sentiment
            sentiment = self._analyze_sentiment(document, processed_text)
            
            # Step 5: Extract topics
            topics = self._extract_topics(document, processed_text)
            
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000
            
            # Create result
            result = ProcessingResult(
                document_id=document.id,
                summary=summary,
                key_points=key_points,
                sentiment=sentiment,
                topics=topics,
                processing_time_ms=processing_time_ms
            )
            
            # Set final attributes
            main_span.set_attribute("processing.total_time_ms", processing_time_ms)
            main_span.set_attribute("result.summary_length", len(summary))
            main_span.set_attribute("result.key_points_count", len(key_points))
            main_span.set_attribute("result.sentiment", sentiment)
            main_span.set_attribute("result.topics_count", len(topics))
            
            # Add result event
            main_span.add_event("processing_completed", {
                "document_id": document.id,
                "processing_time_ms": processing_time_ms,
                "summary_preview": summary[:100] + "..." if len(summary) > 100 else summary
            })
            
            return result
    
    def process_batch(self, documents: List[Document]) -> List[ProcessingResult]:
        """Process a batch of documents."""
        with self.tracer.start_span("batch_processing") as batch_span:
            batch_span.set_attribute("batch.size", len(documents))
            batch_span.set_attribute("batch.id", str(uuid.uuid4()))
            
            results = []
            failed_count = 0
            
            for i, document in enumerate(documents):
                try:
                    with self.tracer.start_span(f"batch_item_{i}") as item_span:
                        item_span.set_attribute("batch.item_index", i)
                        item_span.set_attribute("document.id", document.id)
                        
                        result = self.process_document(document)
                        results.append(result)
                        
                        item_span.set_attribute("item.status", "success")
                        
                except Exception as e:
                    failed_count += 1
                    with self.tracer.start_span(f"batch_item_{i}_error") as error_span:
                        error_span.set_attribute("batch.item_index", i)
                        error_span.set_attribute("document.id", document.id)
                        error_span.record_exception(e)
                        error_span.set_status("error", str(e))
            
            # Set batch summary
            batch_span.set_attribute("batch.processed_count", len(results))
            batch_span.set_attribute("batch.failed_count", failed_count)
            batch_span.set_attribute("batch.success_rate", len(results) / len(documents))
            
            return results
    
    def _preprocess_text(self, document: Document) -> str:
        """Preprocess document text."""
        with self.tracer.start_span("text_preprocessing") as span:
            span.set_attribute("preprocessing.input_length", len(document.content))
            
            # Simulate text cleaning and preprocessing
            time.sleep(0.05)  # Simulate processing time
            
            # Simple preprocessing (in real scenario, this would be more complex)
            processed = document.content.strip()
            processed = " ".join(processed.split())  # Normalize whitespace
            
            span.set_attribute("preprocessing.output_length", len(processed))
            span.set_attribute("preprocessing.compression_ratio", len(processed) / len(document.content))
            
            return processed
    
    def _generate_summary(self, document: Document, text: str) -> str:
        """Generate document summary using LLM."""
        with self.tracer.start_span("summarization") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
            span.set_attribute("gen_ai.operation.name", "summarization")
            span.set_attribute("summarization.input_length", len(text))
            
            # Add input event
            prompt = f"Summarize the following document in 2-3 sentences:\n\n{text[:1000]}..."
            span.add_event("gen_ai.content.prompt", {
                "gen_ai.prompt": prompt
            })
            
            # Simulate LLM call
            time.sleep(0.3)
            
            # Generate summary (simulated)
            summary = self._simulate_summary(text)
            
            # Add output event
            span.add_event("gen_ai.content.completion", {
                "gen_ai.completion": summary
            })
            
            # Set usage metrics
            input_tokens = len(text) // 4
            output_tokens = len(summary) // 4
            
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)
            span.set_attribute("summarization.compression_ratio", len(summary) / len(text))
            
            return summary
    
    def _extract_key_points(self, document: Document, text: str) -> List[str]:
        """Extract key points from document."""
        with self.tracer.start_span("key_point_extraction") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
            span.set_attribute("gen_ai.operation.name", "key_point_extraction")
            
            # Simulate LLM call for key point extraction
            time.sleep(0.2)
            
            # Generate key points (simulated)
            key_points = self._simulate_key_points(text)
            
            span.set_attribute("extraction.key_points_count", len(key_points))
            span.add_event("key_points_extracted", {
                "key_points": key_points
            })
            
            return key_points
    
    def _analyze_sentiment(self, document: Document, text: str) -> str:
        """Analyze document sentiment."""
        with self.tracer.start_span("sentiment_analysis") as span:
            span.set_attribute("analysis.type", "sentiment")
            span.set_attribute("analysis.input_length", len(text))
            
            # Simulate sentiment analysis
            time.sleep(0.1)
            
            # Simple sentiment analysis (simulated)
            sentiment = self._simulate_sentiment(text)
            
            span.set_attribute("analysis.result", sentiment)
            span.add_event("sentiment_analyzed", {
                "sentiment": sentiment,
                "confidence": 0.85  # Simulated confidence
            })
            
            return sentiment
    
    def _extract_topics(self, document: Document, text: str) -> List[str]:
        """Extract topics from document."""
        with self.tracer.start_span("topic_extraction") as span:
            span.set_attribute("analysis.type", "topic_extraction")
            span.set_attribute("analysis.input_length", len(text))
            
            # Simulate topic extraction
            time.sleep(0.15)
            
            # Generate topics (simulated)
            topics = self._simulate_topics(text)
            
            span.set_attribute("extraction.topics_count", len(topics))
            span.add_event("topics_extracted", {
                "topics": topics
            })
            
            return topics
    
    def _simulate_summary(self, text: str) -> str:
        """Simulate summary generation."""
        if "technology" in text.lower():
            return "This document discusses technological advancements and their impact on modern society. It explores various innovations and their potential applications."
        elif "business" in text.lower():
            return "This document covers business strategies and market analysis. It provides insights into industry trends and competitive landscapes."
        else:
            return "This document presents information on various topics with detailed analysis and supporting evidence. It offers comprehensive coverage of the subject matter."
    
    def _simulate_key_points(self, text: str) -> List[str]:
        """Simulate key point extraction."""
        base_points = [
            "Main concept introduction and context",
            "Supporting evidence and examples",
            "Analysis of implications and outcomes"
        ]
        
        if "technology" in text.lower():
            base_points.extend([
                "Technological innovation details",
                "Implementation challenges and solutions"
            ])
        elif "business" in text.lower():
            base_points.extend([
                "Market dynamics and trends",
                "Strategic recommendations"
            ])
        
        return base_points[:4]  # Return up to 4 key points
    
    def _simulate_sentiment(self, text: str) -> str:
        """Simulate sentiment analysis."""
        positive_words = ["good", "great", "excellent", "positive", "success", "benefit"]
        negative_words = ["bad", "poor", "negative", "problem", "issue", "challenge"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _simulate_topics(self, text: str) -> List[str]:
        """Simulate topic extraction."""
        all_topics = [
            "technology", "business", "innovation", "strategy", "analysis",
            "development", "research", "implementation", "management", "growth"
        ]
        
        text_lower = text.lower()
        found_topics = [topic for topic in all_topics if topic in text_lower]
        
        # Return found topics or default topics
        return found_topics[:3] if found_topics else ["general", "information", "analysis"]


def main():
    """Demonstrate document processing with tracing."""
    print("📄 Document Processing Demo")
    print("=" * 40)
    
    # Create processor
    processor = DocumentProcessor()
    
    # Create sample documents
    documents = [
        Document(
            id="doc_001",
            title="AI Technology Trends",
            content="Artificial intelligence technology has been rapidly advancing in recent years. "
                   "Machine learning algorithms are becoming more sophisticated and capable of handling "
                   "complex tasks. The integration of AI in various industries is creating new "
                   "opportunities for innovation and growth. Companies are investing heavily in "
                   "AI research and development to stay competitive in the market.",
            metadata={"author": "Tech Analyst", "category": "technology", "year": 2024}
        ),
        Document(
            id="doc_002", 
            title="Business Strategy Analysis",
            content="Modern business strategy requires careful analysis of market conditions and "
                   "competitive landscapes. Companies must adapt to changing consumer preferences "
                   "and technological disruptions. Strategic planning involves identifying key "
                   "opportunities and potential challenges. Successful businesses focus on "
                   "innovation and customer satisfaction to maintain their market position.",
            metadata={"author": "Business Consultant", "category": "business", "year": 2024}
        ),
        Document(
            id="doc_003",
            title="Research Methodology",
            content="Effective research methodology is crucial for obtaining reliable and valid results. "
                   "Researchers must carefully design their studies to minimize bias and ensure "
                   "accurate data collection. The choice of research methods depends on the nature "
                   "of the research question and available resources. Proper analysis and "
                   "interpretation of data are essential for drawing meaningful conclusions.",
            metadata={"author": "Research Scientist", "category": "research", "year": 2024}
        )
    ]
    
    print(f"Processing {len(documents)} documents...")
    
    # Process documents individually
    print("\n📋 Individual Processing:")
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}: {doc.title}")
        result = processor.process_document(doc)
        
        print(f"  Summary: {result.summary}")
        print(f"  Key Points: {', '.join(result.key_points[:2])}...")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Topics: {', '.join(result.topics)}")
        print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
    
    # Process as batch
    print(f"\n📦 Batch Processing:")
    batch_results = processor.process_batch(documents)
    
    print(f"Processed {len(batch_results)} documents in batch")
    total_time = sum(result.processing_time_ms for result in batch_results)
    print(f"Total processing time: {total_time:.1f}ms")
    print(f"Average time per document: {total_time/len(batch_results):.1f}ms")
    
    # Flush traces
    noveum_trace.flush()
    print(f"\n✅ Traces saved to ./document_traces/")
    
    # Shutdown
    noveum_trace.shutdown()
    
    print("\n🎉 Document processing demo completed!")
    print("Check the trace files to see detailed processing analytics.")


if __name__ == "__main__":
    main()

