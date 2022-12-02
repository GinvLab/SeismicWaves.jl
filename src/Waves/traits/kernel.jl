"""
Abstract trait for a kernel type.
"""
abstract type KernelTypeTrait end

"""
Trait for serial kernel.
"""
struct SerialKernel <: KernelTypeTrait end

"""
Abstract trait for parallel kernel.
"""
abstract type ParallelKernel <: KernelTypeTrait end

"""
Trait for a parallel kernel with multi-threading and shared memory on a single CPU.
"""
struct SharedCPUKernel <: ParallelKernel end

"""
Trait for a parallel kernel with multi-threading and shared memory on a single xPU.
"""
struct SharedxPUKernel <: ParallelKernel end

"""
Trait for a parallel kernel on multiple xPUs.
"""
struct MultixPUKernel <: ParallelKernel end

# Trait constuctor
KernelTypeTrait(x) = KernelTypeTrait(typeof(x))
KernelTypeTrait(x::Type) = error("KernelTypeTrait not implemented for type $(x)")