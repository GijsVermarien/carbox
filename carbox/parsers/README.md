# Unified Chemical Network Parser

The Carbox unified parser provides a single interface for parsing multiple chemical reaction network formats commonly used in astrochemistry.

## Supported Formats

### UCLCHEM Format
- **File format**: CSV with columns for reactants, products, and rate parameters
- **Reaction types**: Gas-phase reactions only (surface chemistry filtered out)
- **Special features**:
  - Supports 3 reactants → 4 products
  - IONOPOL1/IONOPOL2 reactions use KIDA formulas
  - GAR (grain-assisted recombination) reactions
- **Example usage**:
```python
from carbox.parsers import parse_chemical_network
network = parse_chemical_network('data/uclchem_rates.csv', 'uclchem')
```

### UMIST Format
- **File format**: Colon-separated text file
- **Reaction types**: Comprehensive astrochemical reaction types
- **Special features**: Detailed temperature ranges and uncertainty data
- **Example usage**:
```python
network = parse_chemical_network('data/umist22.csv', 'umist')
```

### LATENT-TGAS Format
- **File format**: CSV with simplified 2 reactants → 2 products
- **Reaction types**: Standard Arrhenius kinetics
- **Special features**: Machine learning compatible format
- **Example usage**:
```python
network = parse_chemical_network('data/simple_latent_tgas.csv', 'latent_tgas')
```

## Quick Start

### Basic Usage
```python
from carbox.parsers import parse_chemical_network

# Auto-detect format
network = parse_chemical_network('my_reactions.csv')

# Specify format explicitly
network = parse_chemical_network('my_reactions.csv', 'uclchem')

# Access parsed data
print(f"Species: {len(network.species)}")
print(f"Reactions: {len(network.reactions)}")
```

### Advanced Usage
```python
from carbox.parsers import UnifiedChemicalParser

# Initialize parser
parser = UnifiedChemicalParser()

# List supported formats
print(parser.list_supported_formats())

# Parse with additional options
network = parser.parse('reactions.csv', format_type='uclchem', gas_phase_only=True)
```

## Gas-Phase Filtering (UCLCHEM)

The UCLCHEM parser automatically filters out surface chemistry reactions to focus on gas-phase processes:

- **Excluded reaction types**: FREEZE, DESORB, ER, LH, BULKSWAP, etc.
- **Excluded species**: Any species containing `@` or `#` notation
- **Included types**: PHOTON, CRP, CRPHOT, TWOBODY, IONOPOL1, IONOPOL2, CRS, GAR

## Reaction Type Mapping

Different formats use different reaction type names. The unified parser maps them to consistent Carbox reaction classes:

| Format | Original Type | Carbox Class | Description |
|--------|---------------|--------------|-------------|
| UCLCHEM | PHOTON | FUVReaction | Photodissociation |
| UCLCHEM | CRP | CRPReaction | Cosmic ray proton |
| UCLCHEM | IONOPOL1 | IonPol1Reaction | Ion-polar (KIDA formula 1) |
| UMIST | PD | FUVReaction | Photodissociation |
| UMIST | CR | CRPReaction | Cosmic ray |
| All | Standard | KAReaction | Arrhenius kinetics |

## Rate Parameter Normalization

All formats are normalized to the standard Arrhenius form:
```
k = α × (T/300)^β × exp(-γ/T)
```

| Format | α Parameter | β Parameter | γ Parameter |
|--------|-------------|-------------|-------------|
| UCLCHEM | Alpha | Beta | Gamma |
| UMIST | alpha | beta | gamma |
| LATENT-TGAS | alpha | beta | gamma |

## Error Handling

The parser includes robust error handling:

- **File format detection**: Automatic detection based on filename and structure
- **Missing data**: Graceful handling of missing species or parameters
- **Invalid reactions**: Skip malformed reactions with warnings
- **Type conversion**: Automatic conversion to JAX-compatible arrays

## Integration with Carbox

The unified parser seamlessly integrates with the Carbox framework:

```python
import jax
from carbox.parsers import parse_chemical_network

# Enable JAX configuration for numerical stability
jax.config.update("jax_enable_x64", True)

# Parse network
network = parse_chemical_network('reactions.csv')

# Create JAX-compiled network for simulation
jnetwork = network.compile()

# Run simulation
import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, diffeqsolve

# Define initial conditions
y0 = jnp.array([1e-12] * len(network.species))  # Initial abundances
t0, t1 = 0.0, 1e6  # Time range (years)

# Solve chemical evolution
solution = diffeqsolve(
    ODETerm(jnetwork.dydt),
    solver=Dopri5(),
    t0=t0, t1=t1, dt0=1e-6,
    y0=y0,
    saveat=SaveAt(ts=jnp.logspace(0, 6, 100)),
    args=(jnp.array(100.0), jnp.array(1e-17), jnp.array(1.0), jnp.array(1.0))  # T, cr_rate, uv_field, Av
)
```

## Performance Considerations

- **Sparse matrices**: Use `use_sparse=True` for large networks
- **Vectorization**: Use `vectorize_reactions=True` to group similar reactions
- **JAX compilation**: All rate computations are JAX-compatible for GPU acceleration
- **Memory efficiency**: BCOO sparse format for large reaction networks

## Contributing

To add support for a new format:

1. Create a new parser class inheriting from `BaseParser`
2. Implement `parse_network()` and `parse_reaction()` methods
3. Add format-specific parameter normalization
4. Register the parser with `UnifiedChemicalParser`

Example:
```python
from carbox.parsers import BaseParser, UnifiedChemicalParser

class MyFormatParser(BaseParser):
    def parse_network(self, filepath):
        # Implementation here
        pass

    def parse_reaction(self, row):
        # Implementation here
        pass

# Register the new parser
parser = UnifiedChemicalParser()
parser.register_parser('myformat', MyFormatParser)
```
