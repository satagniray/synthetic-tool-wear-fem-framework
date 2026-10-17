# Synthetic example data

`example_process_fields.csv` is produced by

```bash
python -m tool_wear_demo.cli generate-demo-data
```

It contains one row per wear-surface node of the unworn default tool, ordered
from the flank end, around the rounded cutting edge, to the rake end.

| Column | Unit | Meaning |
|---|---|---|
| `node_id` | - | 1-based node id in the default mesh |
| `surface_type` | - | `flank`, `edge` or `rake` |
| `s_norm` | - | normalised surface coordinate (0 = flank end, 1 = rake end) |
| `d_mm` | mm | signed arc length from the middle of the rounded edge (flank < 0 < rake) |
| `x_mm`, `y_mm` | mm | node position |
| `contact_pressure_MPa` | MPa | synthetic normal contact pressure |
| `temperature_K` | K | synthetic interface temperature |
| `sliding_velocity_m_s` | m/s | synthetic relative sliding velocity |
| `in_contact` | 0/1 | pressure > 0 |
| `wear_rate_um_per_min` | um/min | Usui-type wear rate with the demo constants |

Metadata (`example_process_fields.meta.yaml`):

```yaml
source: synthetic
seed: 42
purpose: portfolio demonstration
not_for_physical_validation: true
```

These values are smooth analytical stand-ins with qualitatively plausible
shapes. They are **not** measured, calibrated or taken from any FEM model and
must not be used for physical predictions.
