#include "tensor2ipc.h"
#include <sys/mman.h>
#include <typeinfo>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

// TODO: This class is not threadsafe.
// We need to use a resource manager to achieve that

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("TensorToIpc")
    .Input("input: T")
    .Attr("T: {float, double}")
    .Attr("address: int")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    });

// CPU specialization of actual computation.
template <typename T>
struct TF2IPCFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, IPCStruct_t* out, const T* in) {
    std::cerr << "Copying out to forces" << size << std::endl;
    std::memcpy(out->mem_handle, in, sizeof(T) * size);
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class TensorToIpcOp : public OpKernel {
 public:
  explicit TensorToIpcOp(OpKernelConstruction* c)
      : OpKernel(c), _output_memory(nullptr) {

    // get memory address
    int64 tmp;
    c->GetAttr("address", &tmp);
    _output_memory = reinterpret_cast<IPCStruct_t*>(tmp);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    if (input.size() > _output_memory->num_elements * _output_memory->element_size ) {
      errors::InvalidArgument(
          "Tensor input size is too large for output buffer!");
    }
    // Do the computation.
    TF2IPCFunctor<Device, T>()(context->eigen_device<Device>(), input.size(),
                               _output_memory, input.data());
  }

 private:
  int _input_size;
  IPCStruct_t* _output_memory;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                              \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("TensorToIpc").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TensorToIpcOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                              \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("TensorToIpc").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      TensorToIpcOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA
