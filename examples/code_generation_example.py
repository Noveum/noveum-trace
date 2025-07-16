"""
Code Generation Assistant Example with Noveum Trace SDK

This example demonstrates how to trace a code generation assistant
that helps developers write, review, and optimize code.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import noveum_trace


class CodeLanguage(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"


class TaskType(Enum):
    """Types of code generation tasks."""

    GENERATE = "generate"
    REVIEW = "review"
    OPTIMIZE = "optimize"
    DEBUG = "debug"
    EXPLAIN = "explain"


@dataclass
class CodeRequest:
    """Represents a code generation request."""

    id: str
    task_type: TaskType
    language: CodeLanguage
    description: str
    existing_code: Optional[str] = None
    requirements: List[str] = None


@dataclass
class CodeResponse:
    """Represents a code generation response."""

    request_id: str
    generated_code: str
    explanation: str
    suggestions: List[str]
    complexity_score: int
    estimated_lines: int
    processing_time_ms: float


class CodeGenerationAssistant:
    """A code generation assistant with comprehensive tracing."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model

        # Initialize tracing
        noveum_trace.init(
            service_name="code-assistant",
            environment="demo",
            log_directory="./code_traces",
            capture_content=True,
        )

        self.tracer = noveum_trace.get_tracer()

    def process_request(self, request: CodeRequest) -> CodeResponse:
        """Process a code generation request."""
        with self.tracer.start_span("code_generation_request") as main_span:
            main_span.set_attribute("request.id", request.id)
            main_span.set_attribute("request.task_type", request.task_type.value)
            main_span.set_attribute("request.language", request.language.value)
            main_span.set_attribute(
                "request.description_length", len(request.description)
            )

            if request.existing_code:
                main_span.set_attribute("request.has_existing_code", True)
                main_span.set_attribute(
                    "request.existing_code_length", len(request.existing_code)
                )
            else:
                main_span.set_attribute("request.has_existing_code", False)

            if request.requirements:
                main_span.set_attribute(
                    "request.requirements_count", len(request.requirements)
                )

            start_time = time.time()

            # Step 1: Analyze request
            analysis = self._analyze_request(request)

            # Step 2: Generate code
            generated_code = self._generate_code(request, analysis)

            # Step 3: Generate explanation
            explanation = self._generate_explanation(request, generated_code)

            # Step 4: Provide suggestions
            suggestions = self._generate_suggestions(request, generated_code)

            # Step 5: Calculate metrics
            complexity_score = self._calculate_complexity(generated_code)
            estimated_lines = len(generated_code.split("\n"))

            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            # Create response
            response = CodeResponse(
                request_id=request.id,
                generated_code=generated_code,
                explanation=explanation,
                suggestions=suggestions,
                complexity_score=complexity_score,
                estimated_lines=estimated_lines,
                processing_time_ms=processing_time_ms,
            )

            # Set final attributes
            main_span.set_attribute("response.code_length", len(generated_code))
            main_span.set_attribute("response.estimated_lines", estimated_lines)
            main_span.set_attribute("response.complexity_score", complexity_score)
            main_span.set_attribute("response.suggestions_count", len(suggestions))
            main_span.set_attribute("processing.total_time_ms", processing_time_ms)

            # Add completion event
            main_span.add_event(
                "request_completed",
                {
                    "request_id": request.id,
                    "task_type": request.task_type.value,
                    "language": request.language.value,
                    "processing_time_ms": processing_time_ms,
                    "code_lines": estimated_lines,
                },
            )

            return response

    def _analyze_request(self, request: CodeRequest) -> Dict[str, Any]:
        """Analyze the code generation request."""
        with self.tracer.start_span("request_analysis") as span:
            span.set_attribute("analysis.task_type", request.task_type.value)
            span.set_attribute("analysis.language", request.language.value)

            # Simulate analysis processing
            time.sleep(0.1)

            # Analyze complexity and requirements
            complexity = self._estimate_task_complexity(request)
            required_tokens = self._estimate_required_tokens(request)

            analysis = {
                "complexity": complexity,
                "required_tokens": required_tokens,
                "has_dependencies": bool(request.requirements),
                "needs_existing_code": bool(request.existing_code),
            }

            span.set_attribute("analysis.complexity", complexity)
            span.set_attribute("analysis.required_tokens", required_tokens)
            span.add_event("analysis_completed", analysis)

            return analysis

    def _generate_code(self, request: CodeRequest, analysis: Dict[str, Any]) -> str:
        """Generate code based on the request."""
        with self.tracer.start_span("code_generation") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", self.model)
            span.set_attribute("gen_ai.operation.name", "code_generation")
            span.set_attribute("generation.language", request.language.value)
            span.set_attribute("generation.task_type", request.task_type.value)

            # Build prompt
            prompt = self._build_code_prompt(request)

            # Add input event
            span.add_event(
                "gen_ai.content.prompt",
                {
                    "gen_ai.prompt": (
                        prompt[:500] + "..." if len(prompt) > 500 else prompt
                    )
                },
            )

            # Simulate LLM processing
            processing_time = 0.5 + (analysis["complexity"] * 0.2)
            time.sleep(processing_time)

            # Generate code (simulated)
            generated_code = self._simulate_code_generation(request)

            # Add output event
            span.add_event(
                "gen_ai.content.completion",
                {
                    "gen_ai.completion": (
                        generated_code[:200] + "..."
                        if len(generated_code) > 200
                        else generated_code
                    )
                },
            )

            # Set usage metrics
            input_tokens = len(prompt) // 4
            output_tokens = len(generated_code) // 4

            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            span.set_attribute(
                "gen_ai.usage.total_tokens", input_tokens + output_tokens
            )
            span.set_attribute("generation.processing_time_ms", processing_time * 1000)

            return generated_code

    def _generate_explanation(self, request: CodeRequest, code: str) -> str:
        """Generate explanation for the generated code."""
        with self.tracer.start_span("explanation_generation") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", self.model)
            span.set_attribute("gen_ai.operation.name", "explanation")

            # Simulate explanation generation
            time.sleep(0.2)

            explanation = self._simulate_explanation(request, code)

            span.set_attribute("explanation.length", len(explanation))
            span.add_event(
                "explanation_generated",
                {
                    "explanation_preview": (
                        explanation[:100] + "..."
                        if len(explanation) > 100
                        else explanation
                    )
                },
            )

            return explanation

    def _generate_suggestions(self, request: CodeRequest, code: str) -> List[str]:
        """Generate improvement suggestions."""
        with self.tracer.start_span("suggestions_generation") as span:
            span.set_attribute("suggestions.language", request.language.value)

            # Simulate suggestion generation
            time.sleep(0.1)

            suggestions = self._simulate_suggestions(request, code)

            span.set_attribute("suggestions.count", len(suggestions))
            span.add_event("suggestions_generated", {"suggestions": suggestions})

            return suggestions

    def _calculate_complexity(self, code: str) -> int:
        """Calculate code complexity score (1-10)."""
        with self.tracer.start_span("complexity_analysis") as span:
            # Simple complexity calculation based on code characteristics
            lines = code.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            # Count complexity indicators
            complexity_indicators = [
                "if",
                "for",
                "while",
                "try",
                "except",
                "class",
                "def",
                "switch",
                "case",
                "catch",
                "function",
                "async",
                "await",
            ]

            indicator_count = sum(
                line.count(indicator)
                for line in non_empty_lines
                for indicator in complexity_indicators
            )

            # Calculate score (1-10)
            complexity_score = min(
                10, max(1, (len(non_empty_lines) + indicator_count) // 5)
            )

            span.set_attribute("complexity.lines_count", len(non_empty_lines))
            span.set_attribute("complexity.indicators_count", indicator_count)
            span.set_attribute("complexity.score", complexity_score)

            return complexity_score

    def _build_code_prompt(self, request: CodeRequest) -> str:
        """Build prompt for code generation."""
        prompt_parts = [
            f"Generate {request.language.value} code for the following task:",
            f"Task: {request.description}",
            f"Type: {request.task_type.value}",
        ]

        if request.existing_code:
            prompt_parts.append(f"Existing code:\n{request.existing_code}")

        if request.requirements:
            prompt_parts.append(f"Requirements: {', '.join(request.requirements)}")

        return "\n\n".join(prompt_parts)

    def _estimate_task_complexity(self, request: CodeRequest) -> int:
        """Estimate task complexity (1-5)."""
        base_complexity = 1

        if request.task_type in [TaskType.OPTIMIZE, TaskType.DEBUG]:
            base_complexity += 2
        elif request.task_type == TaskType.REVIEW:
            base_complexity += 1

        if request.existing_code and len(request.existing_code) > 500:
            base_complexity += 1

        if request.requirements and len(request.requirements) > 3:
            base_complexity += 1

        return min(5, base_complexity)

    def _estimate_required_tokens(self, request: CodeRequest) -> int:
        """Estimate required tokens for the request."""
        base_tokens = len(request.description) // 4

        if request.existing_code:
            base_tokens += len(request.existing_code) // 4

        if request.requirements:
            base_tokens += sum(len(req) for req in request.requirements) // 4

        # Add estimated output tokens
        base_tokens += 200  # Estimated output

        return base_tokens

    def _simulate_code_generation(self, request: CodeRequest) -> str:
        """Simulate code generation based on request."""
        if request.language == CodeLanguage.PYTHON:
            if request.task_type == TaskType.GENERATE:
                return '''def process_data(data):
    """Process the input data and return results."""
    results = []
    for item in data:
        if item is not None:
            processed_item = item.strip().lower()
            results.append(processed_item)
    return results

# Example usage
if __name__ == "__main__":
    sample_data = ["Hello", "World", None, "Python"]
    output = process_data(sample_data)
    print(output)'''
            elif request.task_type == TaskType.REVIEW:
                return '''# Code Review Comments:
# 1. Consider adding type hints for better code clarity
# 2. Add error handling for edge cases
# 3. Use list comprehension for better performance

def process_data(data: List[str]) -> List[str]:
    """Process the input data and return results."""
    try:
        return [item.strip().lower() for item in data if item is not None]
    except AttributeError as e:
        raise ValueError(f"Invalid data format: {e}")'''

        elif request.language == CodeLanguage.JAVASCRIPT:
            return """function processData(data) {
    /**
     * Process the input data and return results
     * @param {Array} data - Input data array
     * @returns {Array} Processed data
     */
    return data
        .filter(item => item !== null && item !== undefined)
        .map(item => item.toString().trim().toLowerCase());
}

// Example usage
const sampleData = ["Hello", "World", null, "JavaScript"];
const output = processData(sampleData);
console.log(output);"""

        else:
            return f"""// Generated {request.language.value} code
// Task: {request.description}
// This is a simulated code generation result"""

    def _simulate_explanation(self, request: CodeRequest, code: str) -> str:
        """Simulate explanation generation."""
        if request.task_type == TaskType.GENERATE:
            return (
                f"This {request.language.value} code implements the requested functionality. "
                f"The main function processes input data according to the specified requirements. "
                f"The code includes proper error handling and follows best practices for {request.language.value}."
            )
        elif request.task_type == TaskType.REVIEW:
            return (
                "The code review identifies several areas for improvement including type safety, "
                "error handling, and performance optimization. The suggested changes will make "
                "the code more robust and maintainable."
            )
        else:
            return (
                f"The {request.task_type.value} operation has been completed for the {request.language.value} code. "
                f"The result addresses the specific requirements mentioned in the request."
            )

    def _simulate_suggestions(self, request: CodeRequest, code: str) -> List[str]:
        """Simulate suggestion generation."""
        base_suggestions = [
            "Add comprehensive error handling",
            "Include unit tests for better reliability",
            "Add documentation and comments",
            "Consider performance optimizations",
        ]

        if request.language == CodeLanguage.PYTHON:
            base_suggestions.extend(
                [
                    "Add type hints for better code clarity",
                    "Use list comprehensions where appropriate",
                    "Follow PEP 8 style guidelines",
                ]
            )
        elif request.language == CodeLanguage.JAVASCRIPT:
            base_suggestions.extend(
                [
                    "Use modern ES6+ features",
                    "Add JSDoc comments",
                    "Consider using TypeScript for type safety",
                ]
            )

        return base_suggestions[:4]  # Return up to 4 suggestions


def main():
    """Demonstrate code generation assistant with tracing."""
    print("💻 Code Generation Assistant Demo")
    print("=" * 40)

    # Create assistant
    assistant = CodeGenerationAssistant()

    # Create sample requests
    requests = [
        CodeRequest(
            id="req_001",
            task_type=TaskType.GENERATE,
            language=CodeLanguage.PYTHON,
            description="Create a function to process and clean text data",
            requirements=[
                "Handle null values",
                "Convert to lowercase",
                "Remove whitespace",
            ],
        ),
        CodeRequest(
            id="req_002",
            task_type=TaskType.REVIEW,
            language=CodeLanguage.PYTHON,
            description="Review existing code for improvements",
            existing_code="def process(data):\n    return [x.lower() for x in data]",
        ),
        CodeRequest(
            id="req_003",
            task_type=TaskType.GENERATE,
            language=CodeLanguage.JAVASCRIPT,
            description="Create a data processing function",
            requirements=["Filter null values", "Transform data"],
        ),
    ]

    print(f"Processing {len(requests)} code generation requests...")

    # Process each request
    for i, request in enumerate(requests, 1):
        print(
            f"\n🔧 Request {i}: {request.task_type.value.title()} ({request.language.value})"
        )
        print(f"Description: {request.description}")

        response = assistant.process_request(request)

        print(f"\n📝 Generated Code ({response.estimated_lines} lines):")
        print("```")
        print(
            response.generated_code[:200] + "..."
            if len(response.generated_code) > 200
            else response.generated_code
        )
        print("```")

        print("\n💡 Explanation:")
        print(
            response.explanation[:150] + "..."
            if len(response.explanation) > 150
            else response.explanation
        )

        print("\n🎯 Suggestions:")
        for suggestion in response.suggestions[:2]:
            print(f"  • {suggestion}")

        print("\n📊 Metrics:")
        print(f"  Complexity Score: {response.complexity_score}/10")
        print(f"  Processing Time: {response.processing_time_ms:.1f}ms")

        time.sleep(0.1)  # Small delay between requests

    # Flush traces
    noveum_trace.flush()
    print("\n✅ Traces saved to ./code_traces/")

    # Shutdown
    noveum_trace.shutdown()

    print("\n🎉 Code generation demo completed!")
    print("Check the trace files to see detailed generation analytics.")


if __name__ == "__main__":
    main()
