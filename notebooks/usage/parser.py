from carbox.parsers import NetworkNames, UnifiedChemicalParser  # noqa

parser = UnifiedChemicalParser()

print(parser.list_supported_formats())

network = parser.parse(
    "reactions.csv", format_type=NetworkNames.uclchem, gas_phase_only=True
)

print(network)
