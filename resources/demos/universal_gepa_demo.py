#!/usr/bin/env python3
"""
Universal GEPA Optimization Demo

This example demonstrates how to use the Universal GEPA optimizer
with different AI frameworks (CrewAI, OpenAI, generic components).
"""

import random
from typing import Dict, Any
from optimas.arch.base import BaseComponent
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction
from optimas.optim.universal_gepa import UniversalGEPAOptimizer


class SummarizationComponent(BaseComponent):
    """A simple text summarization component for demonstration."""
    
    def __init__(self, instruction: str = "Summarize the following text:"):
        super().__init__(
            description="Text summarization component",
            input_fields=["text"],
            output_fields=["summary"],
            variable=instruction
        )
    
    def forward(self, **inputs) -> Dict[str, Any]:
        """Simulate text summarization with instruction prefix."""
        text = inputs.get("text", "")
        
        # Simple simulation: create summary based on instruction style
        if "brief" in self.variable.lower():
            # Brief summary: first sentence + length
            summary = f"{text.split('.')[0]}. (Length: {len(text)} chars)"
        elif "detailed" in self.variable.lower():
            # Detailed summary: more comprehensive
            summary = f"Detailed analysis: {text[:100]}... Key points include main concepts and structure. Total length: {len(text)} characters."
        else:
            # Default summary
            summary = f"Summary: {text[:50]}... (Total: {len(text)} chars)"
        
        return {"summary": summary}


def create_demo_dataset():
    """Create a simple dataset for testing summarization."""
    texts = [
        "Artificial intelligence is transforming the way we work and live. Machine learning algorithms can process vast amounts of data to identify patterns and make predictions.",
        "Climate change is one of the most pressing issues of our time. Rising temperatures, melting ice caps, and extreme weather events are becoming more frequent.",
        "The development of renewable energy sources is crucial for sustainable development. Solar and wind power are becoming increasingly cost-effective alternatives.",
        "Space exploration has led to numerous technological innovations that benefit life on Earth. Satellite technology enables global communications and GPS navigation.",
        "Quantum computing promises to revolutionize computational capabilities. These systems could solve complex problems that are intractable for classical computers."
    ]
    
    examples = []
    for text in texts:
        # Create target summaries (for evaluation)
        target_summary = f"Brief: {text[:30]}..."
        example = Example(text=text, summary=target_summary).with_inputs("text")
        examples.append(example)
    
    return examples


def create_evaluation_metric():
    """Create a simple evaluation metric for summarization quality."""
    def evaluate_summary(gold: Example, pred: Prediction, trace=None) -> float:
        """Evaluate summary quality based on length and content overlap."""
        try:
            gold_summary = gold.summary
            pred_summary = pred.summary
            
            # Simple heuristic evaluation
            score = 0.0
            
            # Length appropriateness (prefer concise summaries)
            pred_length = len(pred_summary)
            if 50 <= pred_length <= 150:
                score += 0.3
            elif pred_length <= 200:
                score += 0.2
            
            # Content overlap (very basic)
            gold_words = set(gold_summary.lower().split())
            pred_words = set(pred_summary.lower().split())
            overlap = len(gold_words & pred_words) / max(len(gold_words), 1)
            score += overlap * 0.4
            
            # Keyword presence
            original_text = gold.text.lower()
            if any(word in pred_summary.lower() for word in ["key", "main", "important", "summary"]):
                score += 0.2
            
            # Structure bonus
            if ":" in pred_summary or "." in pred_summary:
                score += 0.1
                
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0
    
    return evaluate_summary


def create_mock_reflection_lm():
    """Create a mock reflection language model for demo purposes."""
    def mock_reflection_lm(prompt: str) -> str:
        """Generate mock reflection responses based on prompt content."""
        if "improve" in prompt.lower() or "better" in prompt.lower():
            improvements = [
                "Be more concise and focus on key points",
                "Add specific details about the main concepts",
                "Use clearer language and structure",
                "Include brief analysis of important elements",
                "Provide more detailed explanation of core ideas"
            ]
            return random.choice(improvements)
        else:
            return "Focus on creating clear, concise summaries that capture the main ideas."
    
    return mock_reflection_lm


def run_universal_gepa_demo():
    """Run the Universal GEPA optimization demo."""
    print("=" * 60)
    print("Universal GEPA Optimization Demo")
    print("=" * 60)
    
    # Create component and dataset
    print("\n1. Setting up summarization component...")
    component = SummarizationComponent("Summarize the following text:")
    dataset = create_demo_dataset()
    metric = create_evaluation_metric()
    reflection_lm = create_mock_reflection_lm()
    
    print(f"   - Component: {component.__class__.__name__}")
    print(f"   - Initial instruction: '{component.variable}'")
    print(f"   - Dataset size: {len(dataset)} examples")
    
    # Show optimizable components
    print("\n2. Analyzing optimizable components...")
    optimizable = component.gepa_optimizable_components
    print(f"   - Optimizable components: {list(optimizable.keys())}")
    for name, text in optimizable.items():
        print(f"   - {name}: '{text}'")
    
    # Test component before optimization
    print("\n3. Testing component before optimization...")
    test_example = dataset[0]
    result_before = component(text=test_example.text)
    print(f"   - Input: '{test_example.text[:50]}...'")
    print(f"   - Output: '{result_before['summary'][:80]}...'")
    
    # Create and run GEPA optimizer
    print("\n4. Running Universal GEPA optimization...")
    optimizer = UniversalGEPAOptimizer(
        reflection_lm=reflection_lm,
        max_metric_calls=20,  # Small budget for demo
        reflection_minibatch_size=2,
        seed=42
    )
    
    # Run optimization
    result = optimizer.optimize_component(
        component=component,
        trainset=dataset[:3],  # Use subset for faster demo
        metric_fn=metric
    )
    
    # Show results
    print("\n5. Optimization Results:")
    print(f"   - Framework detected: {result.framework_type}")
    print(f"   - Final score: {result.final_score:.3f}")
    print(f"   - Total evaluations: {result.total_evaluations}")
    print(f"   - Optimized components: {result.optimized_components}")
    
    # Show optimized instruction
    optimized_components = component.gepa_optimizable_components
    for name, text in optimized_components.items():
        if name in result.best_candidate:
            print(f"   - {name} (before): '{result.best_candidate[name][:60]}...'")
        print(f"   - {name} (after): '{text[:60]}...'")
    
    # Test component after optimization
    print("\n6. Testing component after optimization...")
    result_after = component(text=test_example.text)
    print(f"   - Input: '{test_example.text[:50]}...'")
    print(f"   - Output: '{result_after['summary'][:80]}...'")
    
    # Compare results
    print("\n7. Performance Comparison:")
    before_score = metric(test_example, Prediction(**result_before))
    after_score = metric(test_example, Prediction(**result_after))
    print(f"   - Score before optimization: {before_score:.3f}")
    print(f"   - Score after optimization: {after_score:.3f}")
    print(f"   - Improvement: {(after_score - before_score):.3f}")
    
    print("\n" + "=" * 60)
    print("Demo completed! The Universal GEPA optimizer can work with")
    print("any BaseComponent across different AI frameworks.")
    print("=" * 60)


if __name__ == "__main__":
    run_universal_gepa_demo()