from resource_estimation.compile_gateset.ppm_compiler.cnot_merging import (
    CNOTMergeTag,
    identify_cnot_merge_groups,
)
from resource_estimation.compile_gateset.ppm_compiler.resource_state_injection import (
    ResourceStateTag,
    replace_resource_gates,
)

__all__ = [
    "CNOTMergeTag",
    "identify_cnot_merge_groups",
    "ResourceStateTag",
    "replace_resource_gates",
]
