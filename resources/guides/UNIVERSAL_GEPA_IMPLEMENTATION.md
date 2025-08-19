# Universal GEPA Implementation in Optimas

This document provides a comprehensive overview of the Universal GEPA implementation that enables GEPA optimization across all AI frameworks supported by Optimas.

## Overview

The Universal GEPA implementation transforms Optimas from having limited DSPy-only GEPA support to a truly framework-agnostic optimization platform where GEPA can optimize any text-based component across any supported AI framework (DSPy, CrewAI, OpenAI, LangChain, etc.).

## Architecture

### Core Components

#### 1. Enhanced BaseComponent (`optimas/arch/base.py`)

Added three key GEPA interface methods to the base class:

```python
@property
def gepa_optimizable_components(self) -> Dict[str, str]:
    """Return mapping of component_name -> optimizable_text for GEPA."""

def apply_gepa_updates(self, updates: Dict[str, str]) -> None:
    """Apply GEPA-optimized text updates to component."""

def extract_execution_trace(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract execution traces for GEPA reflection."""
```

#### 2. Universal GEPA Adapter (`optimas/optim/gepa_adapter.py`)

A framework-agnostic adapter that bridges any BaseComponent with GEPA:

```python
class OptimasGEPAAdapter:
    """Universal GEPA adapter for Optimas BaseComponent optimization."""
    
    def evaluate(self, batch, candidate, capture_traces=False):
        """Evaluate candidate on batch of examples with optional trace capture."""
    
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Create reflective dataset for GEPA optimization."""
```

#### 3. Framework-Specific Feedback Extractors (`optimas/optim/feedback_extractors.py`)

Specialized feedback extraction for each framework:

- `CrewAIFeedbackExtractor`: Extracts agent reasoning, tool usage, role-specific feedback
- `OpenAIFeedbackExtractor`: Analyzes input complexity, function calls, model behavior
- `DSPyFeedbackExtractor`: Signature analysis, reasoning patterns, optimization hints
- `LangChainFeedbackExtractor`: Chain analysis and execution traces

#### 4. Universal GEPA Optimizer (`optimas/optim/universal_gepa.py`)

The main orchestrator that automatically detects framework types and applies appropriate optimization:

```python
class UniversalGEPAOptimizer:
    def optimize_component(self, component, trainset, valset=None, metric_fn=None):
        """Optimize any BaseComponent using appropriate GEPA strategy."""
        
        framework_type = self._detect_framework_type(component)
        
        if framework_type == "dspy":
            return self._optimize_dspy_component(...)
        else:
            return self._optimize_with_universal_adapter(...)
```

#### 5. Enhanced Framework Adapters

Updated existing adapters with GEPA interface implementations:

**CrewAI Adapter (`optimas/adapt/crewai.py`)**:
- Optimizes agent backstory, goal, role, and system messages
- Extracts agent-specific execution traces
- Provides CrewAI-optimized feedback

**OpenAI Adapter (`optimas/adapt/openai.py`)**:
- Optimizes agent instructions and system prompts
- Captures function call information
- Analyzes model behavior patterns

## Framework Detection Logic

The Universal GEPA Optimizer automatically detects framework types using a hierarchical approach:

1. **DSPy**: Has `signature_cls` attribute
2. **CrewAI**: Class name contains "crewai" OR agent has `role` and `backstory`
3. **OpenAI**: Class name contains "openai" OR agent has `instructions` and `model`
4. **LangChain**: Class name contains "langchain"
5. **Generic**: Fallback for custom BaseComponents

## Integration with ComponentOptimizer

The Universal GEPA optimizer is seamlessly integrated into the existing ComponentOptimizer:

```python
# In optimas/optim/cp_optimizer.py
elif self.args.prompt_optimizer == "gepa":
    from optimas.optim.universal_gepa import UniversalGEPAOptimizer
    
    gepa_optimizer = UniversalGEPAOptimizer(
        reflection_lm=self._create_reflection_lm(component),
        # ... other GEPA parameters
    )
    
    result = gepa_optimizer.optimize_component(
        component=component,
        trainset=trainset_per_component,
        metric_fn=metric_from_rm_or_global_metric
    )
```

## Usage Examples

### Basic Usage

```python
from optimas.optim.universal_gepa import UniversalGEPAOptimizer

# Create optimizer
optimizer = UniversalGEPAOptimizer(
    reflection_lm=reflection_lm,
    max_metric_calls=50
)

# Optimize any component
result = optimizer.optimize_component(
    component=your_component,
    trainset=training_examples,
    metric_fn=your_metric
)
```

### With Optimas Configuration

```yaml
# In your config YAML
prompt_optimizer: gepa
gepa_auto: medium  # or set gepa_max_metric_calls, gepa_num_iters, etc.
gepa_reflection_minibatch_size: 5
gepa_log_dir: ./gepa_logs
gepa_use_wandb: true
```

### Framework-Specific Examples

**CrewAI Agent**:
```python
# Component automatically detected as CrewAI
# Optimizes backstory, goal, role
components = crewai_component.gepa_optimizable_components
# Returns: {"backstory": "...", "goal": "...", "role": "..."}
```

**OpenAI Agent**:
```python
# Component automatically detected as OpenAI
# Optimizes instructions, system prompts
components = openai_component.gepa_optimizable_components
# Returns: {"instructions": "..."}
```

**Generic Component**:
```python
# Any BaseComponent with text variables
components = generic_component.gepa_optimizable_components
# Returns: {"ComponentName_text": "..."}
```

## Key Features

### 1. Framework Agnostic
- Works with any BaseComponent regardless of underlying framework
- Automatic framework detection and optimization strategy selection
- Consistent interface across all frameworks

### 2. Rich Feedback Extraction
- Framework-specific feedback extractors provide targeted insights
- Captures execution traces, error patterns, and performance metrics
- Enables more effective reflection and optimization

### 3. Backward Compatibility
- Existing DSPy GEPA integration continues to work unchanged
- No breaking changes to existing Optimas configurations
- Seamless transition from DSPy-only to universal support

### 4. Comprehensive Testing
- 21 comprehensive tests covering all aspects of the implementation
- Framework-specific test scenarios
- Integration tests demonstrating end-to-end functionality

### 5. Extensibility
- Easy to add support for new frameworks
- Pluggable feedback extractor system
- Configurable optimization strategies

## Files Modified/Added

### New Files
- `optimas/optim/gepa_adapter.py` - Universal GEPA adapter
- `optimas/optim/feedback_extractors.py` - Framework-specific feedback extractors
- `optimas/optim/universal_gepa.py` - Universal GEPA optimizer
- `tests/test_universal_gepa.py` - Comprehensive test suite
- `examples/universal_gepa_demo.py` - Demonstration example

### Modified Files
- `optimas/arch/base.py` - Added GEPA interface methods
- `optimas/adapt/crewai.py` - Enhanced with GEPA support
- `optimas/adapt/openai.py` - Enhanced with GEPA support
- `optimas/optim/cp_optimizer.py` - Integrated universal GEPA optimizer

## Benefits

### For Users
1. **Universal Optimization**: Use GEPA with any AI framework, not just DSPy
2. **Better Performance**: Framework-specific feedback leads to more effective optimization
3. **Simplified Configuration**: Same GEPA settings work across all frameworks
4. **Rich Insights**: Detailed optimization logs and framework-specific traces

### For Developers
1. **Extensible Architecture**: Easy to add new frameworks and optimization strategies
2. **Clean Interfaces**: Well-defined protocols for adapters and feedback extractors
3. **Comprehensive Testing**: Robust test coverage ensures reliability
4. **Documentation**: Clear examples and usage patterns

## Future Enhancements

### Phase 2 Possibilities
1. **Multi-Component Optimization**: Optimize multiple components simultaneously
2. **Advanced Merging Strategies**: Framework-aware component merging
3. **Custom Reflection Prompts**: Framework-specific reflection templates
4. **Performance Analytics**: Detailed optimization performance tracking

### Additional Framework Support
1. **LangChain**: Full integration with chain optimization
2. **AutoGen**: Multi-agent system optimization
3. **Custom Frameworks**: Template for adding new framework support

## Conclusion

The Universal GEPA implementation represents a significant advancement in Optimas' optimization capabilities. By providing framework-agnostic GEPA support with rich, framework-specific feedback, it enables users to leverage the power of GEPA optimization regardless of their chosen AI framework. The implementation maintains backward compatibility while opening up new possibilities for cross-framework optimization strategies.

This implementation transforms Optimas from a DSPy-centric optimization tool into a truly universal platform for optimizing compound AI systems across any supported framework.