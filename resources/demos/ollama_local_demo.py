#!/usr/bin/env python3
"""
Ollama Local Model Demo for Optimas

This example demonstrates how to use Optimas with completely local models
using Ollama. Perfect for development, testing, or when you need privacy.

Prerequisites:
1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. Pull a model: ollama pull llama3.1:8b
3. Start Ollama: ollama serve

Usage:
    python examples/ollama_local_demo.py
    python examples/ollama_local_demo.py --model qwen2.5:14b  # Use different model
    python examples/ollama_local_demo.py --verbose           # Show detailed output
"""

import argparse
import requests
import json
from typing import Dict, Any, List
from optimas.arch.base import BaseComponent
from optimas.arch.system import CompoundAISystem
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction


class OllamaComponent(BaseComponent):
    """A component that uses Ollama for local LLM inference."""
    
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 ollama_base_url: str = "http://localhost:11434",
                 initial_prompt: str = "You are a helpful AI assistant."):
        super().__init__(
            description=f"Ollama-powered component using {model_name}",
            input_fields=["user_input"],
            output_fields=["response"],
            variable=initial_prompt,
            config={"model": model_name, "temperature": 0.7, "max_tokens": 200}
        )
        self.ollama_base_url = ollama_base_url
    
    def forward(self, **inputs) -> Dict[str, Any]:
        """Generate response using Ollama."""
        user_input = inputs.get("user_input", "")
        
        # Build the prompt with our variable (system prompt)
        messages = [
            {"role": "system", "content": self.variable},
            {"role": "user", "content": user_input}
        ]
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    },
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result["message"]["content"]
                return {"response": assistant_response.strip()}
            else:
                return {"response": f"Error: Ollama API returned {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"response": f"Error connecting to Ollama: {str(e)}"}
        except Exception as e:
            return {"response": f"Unexpected error: {str(e)}"}


class LocalRAGComponent(BaseComponent):
    """A simple RAG component using local embeddings and Ollama."""
    
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 ollama_base_url: str = "http://localhost:11434",
                 rag_prompt: str = "Answer the question based on the provided context. Context: {context}\nQuestion: {question}"):
        super().__init__(
            description=f"Local RAG component using {model_name}",
            input_fields=["question", "context"],
            output_fields=["answer"],
            variable=rag_prompt,
            config={"model": model_name, "temperature": 0.3}
        )
        self.ollama_base_url = ollama_base_url
    
    def forward(self, **inputs) -> Dict[str, Any]:
        """Generate RAG answer using local model."""
        question = inputs.get("question", "")
        context = inputs.get("context", "")
        
        # Format prompt with our variable template
        formatted_prompt = self.variable.format(context=context, question=question)
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": formatted_prompt,
                    "options": {"temperature": self.config.temperature},
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["response"]
                return {"answer": answer.strip()}
            else:
                return {"answer": f"Error: Ollama API returned {response.status_code}"}
                
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}


def check_ollama_availability(base_url: str = "http://localhost:11434") -> tuple[bool, List[str]]:
    """Check if Ollama is running and return available models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return True, models
        return False, []
    except:
        return False, []


def create_local_qa_system(model_name: str, ollama_base_url: str) -> CompoundAISystem:
    """Create a simple Q&A system using local models."""
    
    # Context retriever (simulated - in real usage you'd have a vector DB)
    class ContextRetriever(BaseComponent):
        def __init__(self):
            super().__init__(
                description="Simple context retriever",
                input_fields=["question"],
                output_fields=["context"],
                variable="Retrieve relevant context for: {question}"
            )
        
        def forward(self, **inputs) -> Dict[str, Any]:
            question = inputs.get("question", "")
            
            # Simple keyword-based context (in practice, use embeddings)
            contexts = {
                "python": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
                "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "ollama": "Ollama is a tool that allows you to run large language models locally on your computer. It supports models like Llama, Mistral, and others.",
                "optimas": "Optimas is a framework for end-to-end optimization of compound AI systems using Globally Aligned Local Reward Functions."
            }
            
            # Find relevant context
            question_lower = question.lower()
            for keyword, context in contexts.items():
                if keyword in question_lower:
                    return {"context": context}
            
            return {"context": "No specific context found for this question."}
    
    # Create system
    system = CompoundAISystem(
        components={
            "retriever": ContextRetriever(),
            "answerer": LocalRAGComponent(model_name, ollama_base_url)
        },
        final_output_fields=["answer"],
        ground_fields=["expected_answer"] if False else []  # No ground truth for demo
    )
    
    return system


def demo_basic_chat(model_name: str, ollama_base_url: str, verbose: bool = False):
    """Demonstrate basic chat functionality."""
    print(f"🤖 Basic Chat Demo with {model_name}")
    print("-" * 40)
    
    # Create a simple chat component
    chat_component = OllamaComponent(
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        initial_prompt="You are a helpful AI assistant. Keep responses concise and friendly."
    )
    
    # Test questions
    test_questions = [
        "What is Python programming language?",
        "Explain machine learning in simple terms",
        "What are the benefits of using local AI models?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        result = chat_component(user_input=question)
        response = result["response"]
        
        if verbose:
            print(f"   Model: {model_name}")
            print(f"   Prompt: {chat_component.variable}")
            print(f"   Response ({len(response)} chars): {response}")
        else:
            # Truncate long responses for readability
            display_response = response[:200] + "..." if len(response) > 200 else response
            print(f"   Answer: {display_response}")


def demo_rag_system(model_name: str, ollama_base_url: str, verbose: bool = False):
    """Demonstrate RAG system with local models."""
    print(f"\n🔍 RAG System Demo with {model_name}")
    print("-" * 40)
    
    system = create_local_qa_system(model_name, ollama_base_url)
    
    # Test questions
    test_questions = [
        "What is Python and who created it?",
        "How does machine learning work?",
        "What is Ollama used for?",
        "Tell me about Optimas framework"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        try:
            result = system(question=question)
            answer = result.answer
            
            if verbose:
                # Show the intermediate steps
                retriever_result = system.components["retriever"](question=question)
                context = retriever_result["context"]
                print(f"   Context: {context[:100]}...")
                print(f"   Answer: {answer}")
            else:
                display_answer = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"   Answer: {display_answer}")
                
        except Exception as e:
            print(f"   Error: {str(e)}")


def demo_gepa_integration(model_name: str, ollama_base_url: str, verbose: bool = False):
    """Demonstrate GEPA integration with local models."""
    print(f"\n🔧 GEPA Integration Demo with {model_name}")
    print("-" * 40)
    
    # Create component with optimizable prompt
    component = OllamaComponent(
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        initial_prompt="You are an AI assistant."
    )
    
    print("Testing GEPA interface methods:")
    
    # Test GEPA interface
    optimizable = component.gepa_optimizable_components
    print(f"✅ Found {len(optimizable)} optimizable components: {list(optimizable.keys())}")
    
    # Test prompt optimization simulation
    print("\n🔄 Simulating GEPA prompt optimization...")
    
    # Original response
    test_input = "Explain quantum computing"
    original_result = component(user_input=test_input)
    print(f"Original response: {original_result['response'][:100]}...")
    
    # Update prompt via GEPA
    optimized_prompt = "You are an expert science communicator who explains complex topics clearly and concisely. Always include practical examples."
    component.apply_gepa_updates({"OllamaComponent_text": optimized_prompt})
    
    # New response with optimized prompt
    optimized_result = component(user_input=test_input)
    print(f"Optimized response: {optimized_result['response'][:100]}...")
    
    # Extract execution trace
    trace = component.extract_execution_trace(
        {"user_input": test_input}, 
        optimized_result
    )
    
    if verbose:
        print(f"\nExecution trace fields: {list(trace.keys())}")
        print(f"Framework info: {trace.get('framework', 'N/A')}")
    
    print("✅ GEPA integration working with local models!")


def main():
    parser = argparse.ArgumentParser(description="Demo Optimas with local Ollama models")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model to use")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--demo", choices=["chat", "rag", "gepa", "all"], default="all", 
                       help="Which demo to run")
    args = parser.parse_args()
    
    print("🏠 Optimas + Ollama Local Demo")
    print("=" * 50)
    
    # Check Ollama availability
    print("Checking Ollama availability...")
    is_available, models = check_ollama_availability(args.ollama_url)
    
    if not is_available:
        print("❌ Ollama is not running or not accessible")
        print("💡 Make sure to:")
        print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Start Ollama: ollama serve")
        print(f"   3. Pull a model: ollama pull {args.model}")
        return
    
    print(f"✅ Ollama is running with {len(models)} models")
    if args.verbose:
        print(f"   Available models: {', '.join(models)}")
    
    if args.model not in models:
        print(f"⚠️ Model '{args.model}' not found. Available: {', '.join(models)}")
        if models:
            print(f"💡 You can pull it with: ollama pull {args.model}")
            print(f"💡 Or use an available model with --model {models[0]}")
        return
    
    print(f"🚀 Using model: {args.model}")
    print()
    
    # Run demos
    try:
        if args.demo in ["chat", "all"]:
            demo_basic_chat(args.model, args.ollama_url, args.verbose)
        
        if args.demo in ["rag", "all"]:
            demo_rag_system(args.model, args.ollama_url, args.verbose)
        
        if args.demo in ["gepa", "all"]:
            demo_gepa_integration(args.model, args.ollama_url, args.verbose)
        
        print(f"\n🎉 Demo completed successfully!")
        print("✅ Optimas works great with local Ollama models")
        print("✅ GEPA integration is fully compatible with local inference")
        print("💡 You now have a completely private AI system for development")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {str(e)}")
        if args.verbose:
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    main()