# Python SDK

DeepFabric provides a Python SDK for programmatic dataset generation. This is useful for integration into automated pipelines, Jupyter notebooks, or custom workflows.

## Installation

```bash
pip install deepfabric
```

## Quick Start

```python
from deepfabric import DeepFabricSDK

# Initialize the SDK
sdk = DeepFabricSDK(verbose=True)

# Generate a dataset from a config file
result = sdk.generate("config.yaml")

if result.success:
    print(f"Generated {result.samples_generated} samples")
    print(f"Dataset saved to: {result.dataset_path}")
    print(f"Duration: {result.duration_seconds:.1f}s")
else:
    print(f"Generation failed: {result.error}")
```

## DeepFabricSDK Class

### Constructor

```python
DeepFabricSDK(*, verbose: bool = False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | bool | `False` | Enable verbose logging output |

### generate() Method

```python
sdk.generate(
    config_path: str | Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    num_samples: int | str | None = None,
    batch_size: int | None = None,
    topic_only: bool = False,
) -> GenerationResult
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | str \| Path | Path to YAML configuration file |
| `provider` | str | Override provider from config |
| `model` | str | Override model from config |
| `temperature` | float | Override temperature from config |
| `num_samples` | int \| str | Number of samples to generate |
| `batch_size` | int | Batch size for generation |
| `topic_only` | bool | Only generate topic structure |

## GenerationResult

The `generate()` method returns a `GenerationResult` dataclass:

```python
@dataclass
class GenerationResult:
    success: bool                    # Whether generation succeeded
    dataset_path: str | None         # Path to generated dataset
    topic_path: str | None           # Path to topic structure
    samples_generated: int           # Number of successful samples
    samples_failed: int              # Number of failed samples
    duration_seconds: float          # Total generation time
    error: str | None                # Error message if failed
    metadata: dict[str, Any]         # Additional metadata
```

### Properties

| Property | Description |
|----------|-------------|
| `total_samples` | Total samples attempted (generated + failed) |

## Examples

### Basic Usage

```python
from deepfabric import DeepFabricSDK

sdk = DeepFabricSDK()
result = sdk.generate("config.yaml")

print(f"Success: {result.success}")
print(f"Samples: {result.samples_generated}")
print(f"Duration: {result.duration_seconds}s")
```

### With Provider Override

```python
from deepfabric import DeepFabricSDK

sdk = DeepFabricSDK()
result = sdk.generate(
    "config.yaml",
    provider="azure",
    model="gpt-4o-deployment",
    num_samples=100,
)

if result.success:
    print(f"Generated with Azure: {result.samples_generated} samples")
```

### Error Handling

```python
from deepfabric import DeepFabricSDK

sdk = DeepFabricSDK()
result = sdk.generate("config.yaml")

if not result.success:
    print(f"Error: {result.error}")
    # Handle error appropriately
else:
    # Process successful result
    print(f"Dataset at: {result.dataset_path}")
```

### Topic-Only Generation

Generate just the topic structure without creating samples:

```python
from deepfabric import DeepFabricSDK

sdk = DeepFabricSDK()
result = sdk.generate("config.yaml", topic_only=True)

if result.success:
    print(f"Topic structure saved to: {result.topic_path}")
```

### Using with Azure OpenAI

```python
import os
from deepfabric import DeepFabricSDK

# Set Azure credentials
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com"

# Create config with Azure provider
config_yaml = """
llm:
  provider: azure
  model: gpt-4o-deployment
  temperature: 0.7

topics:
  prompt: "Machine learning concepts"
  mode: tree
  depth: 2
  degree: 3
  save_as: topics.jsonl

generation:
  system_prompt: "Generate educational examples"
  conversation:
    type: basic

output:
  num_samples: 10
  save_as: dataset.jsonl
"""

# Write config and generate
with open("azure_config.yaml", "w") as f:
    f.write(config_yaml)

sdk = DeepFabricSDK(verbose=True)
result = sdk.generate("azure_config.yaml")

print(f"Generated {result.samples_generated} samples with Azure")
```

## Comparison: SDK vs CLI

| Feature | CLI | SDK |
|---------|-----|-----|
| Usage | Command line | Python code |
| Output | Files + terminal | `GenerationResult` object |
| Error handling | Exit codes | Exception or result.error |
| Integration | Shell scripts | Python applications |
| Progress display | TUI | Logging |

## See Also

- [Configuration Reference](dataset-generation/configuration.md) - YAML configuration options
- [CLI Reference](cli/index.md) - Command-line interface
- [Basic Datasets](dataset-generation/basic.md) - Dataset generation guide
