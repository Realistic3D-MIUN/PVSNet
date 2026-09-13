import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print("Status: Loading TensorRT engine from: ", engine_path)
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Status: Engine file not found at {engine_path}.")
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.inputs = {}
        self.outputs = {}
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape)
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            self.context.set_tensor_address(tensor_name, int(device_mem))

            binding_data = {
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype,
                'size': size
            }

            if mode == trt.TensorIOMode.INPUT:
                self.inputs[tensor_name] = binding_data
            else:
                self.outputs[tensor_name] = binding_data

    def infer(self, feed_dict):
        for name, data in feed_dict.items():
            if name not in self.inputs:
                raise ValueError(f"Input name {name} not found in engine inputs.")
            
            target_dtype = self.inputs[name]['dtype']
            data_flat = np.ascontiguousarray(data.flatten().astype(target_dtype))
            
            expected_size = self.inputs[name]['size']
            if data_flat.size != expected_size:
                raise ValueError(f"Input {name} size mismatch. Expected {expected_size}, got {data_flat.size}")
            
            np.copyto(self.inputs[name]['host'], data_flat)
            cuda.memcpy_htod_async(self.inputs[name]['device'], self.inputs[name]['host'], self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for name in self.outputs:
            cuda.memcpy_dtoh_async(self.outputs[name]['host'], self.outputs[name]['device'], self.stream)
        
        self.stream.synchronize()

        results = {}
        for name, data in self.outputs.items():
            results[name] = data['host'].reshape(data['shape'])
        
        return results

