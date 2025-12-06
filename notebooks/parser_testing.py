from carbox.parsers import UnifiedChemicalParser

# Initialize parser
parser = UnifiedChemicalParser()

# List supported formats
print(parser.list_supported_formats())

# Parse with additional options
network = parser.parse("reactions.csv", format_type="uclchem", gas_phase_only=True)
