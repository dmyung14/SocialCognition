from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RunOptions:
    render_format: str = "mp4"
    device: str = "auto"
    debug: bool = False
    analysis_fps: float = 15.0
    sync_threshold: float = 0.5
    model_dir: str = "assets/models"
    force: bool = False
    inventory_only: bool = False

    def __post_init__(self) -> None:
        if self.render_format not in {"mp4", "gif", "both"}:
            raise ValueError("render_format must be mp4, gif, or both")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if not math.isfinite(self.analysis_fps) or not 0 < self.analysis_fps <= 120:
            raise ValueError("analysis_fps must be finite and in (0, 120]")
        if not math.isfinite(self.sync_threshold) or not 0 <= self.sync_threshold <= 1:
            raise ValueError("sync_threshold must be in [0, 1]")
