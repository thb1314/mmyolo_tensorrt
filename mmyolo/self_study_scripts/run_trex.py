import matplotlib.pyplot as plt
import os
import pandas as pd
from trex import *

# Configure a wider output (for the wide graphs)
set_wide_display()

plan = EnginePlan('./layer.json', './profile.json', './profile.metadata.json')

formatter = layer_type_formatter if True else precision_formatter
graph = to_dot(plan, formatter)
svg_name = render_dot(graph, 'demo.engine', 'svg')