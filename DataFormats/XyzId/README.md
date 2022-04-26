## Define the alpaka-based SoA data formats

Notes:
  - do not define a dictionary for `XyzIdHostCollection`, because it is the
    same class as `alpaka_serial_sync::XyzIdDeviceCollection`;
  - define the dictionary for `alpaka_cuda_async::XyzIdDeviceCollection` as
    _transient_ only;
  - the dictionary for `alpaka_cuda_async::XyzIdDeviceCollection` should be
    defined in a separate library, to factor out the CUDA dependency.
