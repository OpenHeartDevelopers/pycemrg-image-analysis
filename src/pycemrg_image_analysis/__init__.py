# src/pycemrg_image_analysis/__init__.py

__all__ = ["ImageAnalysisScaffolder"]


def __getattr__(name):
    if name == "ImageAnalysisScaffolder":
        from pycemrg_image_analysis.scaffolding import ImageAnalysisScaffolder
        return ImageAnalysisScaffolder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")