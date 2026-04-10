"""Model diagnostics and ranking visualizations for the alpha ranking pipeline."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.model.model import AlphaXGBoost
