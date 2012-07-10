__global__ void set_one(float *array, int i) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id == 0) {
        array[i] = 1.0f;
    }
}

__device__ float gpu_logistic(float x) {
    return 1 / (1 + expf(-x));
}

__global__ void activation_function(float x[], int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n){
        x[id] = gpu_logistic(x[id]);
    }
}
