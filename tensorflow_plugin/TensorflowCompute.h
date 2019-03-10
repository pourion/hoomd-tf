// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause
// License.


#ifndef _TENSORFLOW_COMPUTE_H_
#define _TENSORFLOW_COMPUTE_H_

/*! \file TensorflowCompute.h
    \brief Declaration of TensorflowCompute
*/

#include <hoomd/Autotuner.h>
#include <hoomd/ForceCompute.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.h>
#include <hoomd/SystemDefinition.h>
#include <hoomd/md/NeighborList.h>
#include "TFArrayComm.h"
#include "TaskLock.h"
#include "TensorflowCompute.h"

// pybind11 is used to create the python bindings to the C++ object,
// but not if we are compiling GPU kernels
#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#include <hoomd/extern/pybind/include/pybind11/stl_bind.h>
#endif


namespace hoomd_tf {


  //! A nonsense particle Compute written to demonstrate how to write a plugin
  /*! This Compute simply sets all of the particle's velocities to 0 when update()
  * is called.
  */

 /*!
 * Indicates if forces should be computed by or passed to TF
 */
  enum class FORCE_MODE { tf2hoomd, hoomd2tf };

  /*! Template class for TFCompute
  *  \tfparam M If TF is on CPU or GPU.
  *
  */
  template <TFCommMode M = TFCommMode::CPU>
  class TensorflowCompute : public ForceCompute {
  public:
    //! Constructor
    TensorflowCompute(pybind11::object& py_self,
                      std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<NeighborList> nlist, Scalar r_cut,
                      unsigned int nneighs, FORCE_MODE force_mode,
                      unsigned int period,
                      TaskLock* tasklock);

    TensorflowCompute() = delete;

    //! Destructor
    virtual ~TensorflowCompute();

    Scalar getLogValue(const std::string& quantity,
                      unsigned int timestep) override;

    int64_t getForcesBuffer() const;
    int64_t getPositionsBuffer() const;
    int64_t getVirialBuffer() const;
    int64_t getNlistBuffer() const;

    bool isDoublePrecision() const {
  #ifdef SINGLE_PRECISION
      return false;
  #else
      return true;
  #endif  // SINGLE_PRECISION
    }

    std::vector<Scalar4> getForcesArray() const;
    std::vector<Scalar4> getNlistArray() const;
    std::vector<Scalar4> getPositionsArray() const;
    std::vector<Scalar> getVirialArray() const;
    unsigned int getVirialPitch() const { return m_virial.getPitch(); }

    pybind11::object
        _py_self;  // pybind objects have to be public with current cc flags

  protected:
    // used if particle number changes
    virtual void reallocate();
    //! Take one timestep forward
    virtual void computeForces(unsigned int timestep) override;

    virtual void prepareNeighbors();
    virtual void receiveVirial();

    void finishUpdate(unsigned int timestep);

    std::shared_ptr<NeighborList> m_nlist;
    Scalar _r_cut;
    unsigned int _nneighs;
    FORCE_MODE _force_mode;
    unsigned int _period;
    std::string m_log_name;
    TaskLock* _tasklock;

    TFArrayComm<M, Scalar4> _positions_comm;
    TFArrayComm<M, Scalar4> _forces_comm;
    GPUArray<Scalar4> _nlist_array;
    GPUArray<Scalar> _virial_array;
    TFArrayComm<M, Scalar4> _nlist_comm;
    TFArrayComm<M, Scalar> _virial_comm;
  };

  //! Export the TensorflowCompute class to python
  void export_TensorflowCompute(pybind11::module& m);


  #ifdef ENABLE_CUDA

  class TensorflowComputeGPU : public TensorflowCompute<TFCommMode::GPU> {
  public:
    //! Constructor
    TensorflowComputeGPU(pybind11::object& py_self,
                        std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<NeighborList> nlist, Scalar r_cut,
                        unsigned int nneighs, FORCE_MODE force_mode,
                        unsigned int period,
                        TaskLock* tasklock);

    void setAutotunerParams(bool enable, unsigned int period) override;

  protected:
    void computeForces(unsigned int timestep) override;
    void reallocate() override;
    void prepareNeighbors() override;
    void receiveVirial() override;

  private:
    std::unique_ptr<Autotuner> m_tuner;  // Autotuner for block size
    cudaStream_t _streams[4];
    size_t _nstreams = 4;
  };

  //! Export the TensorflowComputeGPU class to python
  void export_TensorflowComputeGPU(pybind11::module& m);

  template class TensorflowCompute<TFCommMode::GPU>;
  #endif  // ENABLE_CUDA

  // force implementation
  template class TensorflowCompute<TFCommMode::CPU>;

}

#endif  // _TENSORFLOW_COMPUTE_H_
