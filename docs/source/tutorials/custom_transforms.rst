Custom Transforms Tutorial
===========================

Create custom data transformations.

Basic Custom Transform
-----------------------

::

    from quantiq.transform import DatasetTransform
    from quantiq.data import OneDimensionalDataset

    class MyTransform(DatasetTransform):
        def __init__(self, param1: float):
            self.param1 = param1

        def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            new_y = dataset.y * self.param1
            return dataset.copy_with(y=new_y)

Use in Pipeline
---------------

::

    from quantiq.transform import Pipeline

    pipeline = Pipeline([
        MyTransform(param1=2.0),
        GaussianSmoothing(sigma=1.0)
    ])

See :doc:`../user_guide/concepts` for transform architecture.
