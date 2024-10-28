def __getattr__(name):
    if name == "EGL":
        from .egl import EGL

        return EGL
    if name == "TRCovarianceMatrixAdaptation":
        from .cma import TRCovarianceMatrixAdaptation

        return TRCovarianceMatrixAdaptation
    if name == "IGL":
        from .igl import IGL

        return IGL
    if name == "minimize":
        from .alg_optimizer import minimize

        return minimize
    raise AttributeError(f"module {__name__} has no attribute {name}")
