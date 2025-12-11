# Docker Configuration

This directory contains Docker configurations for the MNIST Neural Network project.

## Available Images

### Dockerfile.cpu
**Purpose:** Development and CPU-only builds  
**Base:** Ubuntu 22.04  
**Use Case:** Development, testing, sanitizers, code analysis

**Includes:**
- ✅ GCC 13 and Clang compilers (C++20 support)
- ✅ CMake and Ninja build system
- ✅ clang-format and clang-tidy
- ✅ GDB debugger
- ✅ Valgrind memory profiler
- ✅ lcov 2.1 with Perl dependencies for coverage
- ✅ AddressSanitizer, UBSan, TSan libraries
- ✅ All development tools

### Dockerfile.gpu
**Purpose:** Production builds with CUDA support  
**Base:** nvidia/cuda:12.2.2-devel-ubuntu22.04  
**Use Case:** GPU training, CUDA development, production deployment

**Includes:**
- ✅ All CPU development tools
- ✅ CUDA Toolkit 12.2.2
- ✅ NVIDIA compiler (nvcc)
- ✅ CUDA profiling tools (Nsight Systems, cuda-memcheck)
- ✅ Automatic build of release and debug versions

## Quick Start

### CPU Development Container

```bash
# Build
docker build -t mnist-cpu -f docker/Dockerfile.cpu .

# Run with source mounted (recommended for development)
docker run -it -v $(pwd):/workspace mnist-cpu bash

# Inside container
./build.sh debug
./build.sh asan
./build.sh test
```

### GPU Production Container

```bash
# Build
docker build -t mnist-gpu -f docker/Dockerfile.gpu .

# Run (pre-built executable)
docker run --gpus all mnist-gpu

# Run with interactive shell
docker run --gpus all -it mnist-gpu bash

# Profile GPU execution
docker run --gpus all mnist-gpu nsys profile --stats=true ./build/mnist_net
```

## Development Workflows

### 1. Format Code

```bash
# CPU container
docker run -v $(pwd):/workspace mnist-cpu bash -c "./build.sh format-fix"
```

### 2. Static Analysis

```bash
# CPU container
docker run -v $(pwd):/workspace mnist-cpu bash -c "./build.sh tidy"
```

### 3. Memory Error Detection

```bash
# CPU container with AddressSanitizer
docker run -v $(pwd):/workspace mnist-cpu bash -c "./build.sh asan && ./build-asan/mnist_net"
```

### 4. Run Tests

```bash
# CPU container
docker run -v $(pwd):/workspace mnist-cpu bash -c "./build.sh test"

# GPU container
docker run --gpus all mnist-gpu bash -c "./build.sh test"
```

### 5. GPU Profiling

```bash
# GPU container with Nsight Systems
docker run --gpus all mnist-gpu nsys profile --stats=true ./build-release/mnist_net

# GPU memory checking
docker run --gpus all mnist-gpu cuda-memcheck ./build/mnist_net
```

## Container Comparison

| Feature | CPU Container | GPU Container |
|---------|---------------|---------------|
| **Build Speed** | Fast | Slower (larger base image) |
| **Size** | ~500MB | ~7GB |
| **CUDA Support** | ❌ | ✅ |
| **Sanitizers** | ✅ Full support | ⚠️ CPU code only |
| **Debugging** | ✅ GDB, Valgrind | ✅ GDB, cuda-gdb |
| **Profiling** | ✅ perf, valgrind | ✅ + Nsight, nvprof |
| **Use Case** | Development, CI/CD | Training, Production |

## Advanced Usage

### Bind Mount for Development

```bash
# Changes in container reflect on host immediately
docker run -it -v $(pwd):/workspace \
  -v $(pwd)/build:/workspace/build \
  mnist-cpu bash
```

### Run with Custom Arguments

```bash
# GPU with memory limits
docker run --gpus all -m 4g mnist-gpu ./build-release/mnist_net --epochs 10

# CPU with CPU limit
docker run --cpus=2 mnist-cpu bash -c "./build.sh release"
```

### Multi-Stage Build for Smaller Production Image

```dockerfile
# Example: Create smaller production image from GPU build
FROM mnist-gpu as builder

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
COPY --from=builder /app/build-release/mnist_net /app/mnist_net
COPY --from=builder /app/data /app/data
CMD ["/app/mnist_net"]
```

### Using with Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  dev:
    build:
      context: .
      dockerfile: docker/Dockerfile.cpu
    volumes:
      - .:/workspace
    command: bash
    stdin_open: true
    tty: true

  gpu:
    build:
      context: .
      dockerfile: docker/Dockerfile.gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command: ./build-release/mnist_net
```

Then run:
```bash
docker-compose up dev    # CPU development
docker-compose up gpu    # GPU training
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t mnist-test -f docker/Dockerfile.cpu .
      
      - name: Run tests
        run: docker run mnist-test bash -c "./build.sh test"
      
      - name: Check formatting
        run: docker run mnist-test bash -c "./build.sh format-check"
      
      - name: Run static analysis
        run: docker run mnist-test bash -c "./build.sh tidy"
      
      - name: Test with sanitizers
        run: docker run mnist-test bash -c "./build.sh asan && ./build-asan/tests/unit_tests"
```

## Troubleshooting

### GPU Container Issues

**Problem:** "docker: Error response from daemon: could not select device driver"

**Solution:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Problem:** "could not find compatible CUDA version"

**Solution:** Check your NVIDIA driver version matches CUDA requirements:
```bash
nvidia-smi  # Check driver version
# Update Dockerfile.gpu to use appropriate CUDA base image
```

### Build Performance

**Problem:** Slow Docker builds

**Solutions:**
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t mnist-cpu -f docker/Dockerfile.cpu .

# Use build cache from registry
docker pull myregistry/mnist-cpu:latest || true
docker build --cache-from myregistry/mnist-cpu:latest -t mnist-cpu -f docker/Dockerfile.cpu .
```

### Permissions Issues

**Problem:** Files created in container owned by root

**Solutions:**
```bash
# Run with current user ID
docker run -u $(id -u):$(id -g) -v $(pwd):/workspace mnist-cpu bash

# Or change ownership after
docker run -v $(pwd):/workspace mnist-cpu bash -c "chown -R $(id -u):$(id -g) /workspace/build"
```

## Best Practices

1. **Use CPU container for development** - faster, smaller, supports all sanitizers
2. **Use GPU container for training** - production-ready, optimized builds
3. **Mount source code** - don't copy during development
4. **Use .dockerignore** - exclude build artifacts to speed up builds
5. **Layer caching** - order Dockerfile commands from least to most frequently changing
6. **Multi-stage builds** - create smaller production images
7. **Tag images** - version your Docker images for reproducibility

## Image Registry

Push to a registry for team sharing:

```bash
# Tag images
docker tag mnist-cpu myregistry/mnist-cpu:latest
docker tag mnist-gpu myregistry/mnist-gpu:latest

# Push
docker push myregistry/mnist-cpu:latest
docker push myregistry/mnist-gpu:latest

# Pull on other machines
docker pull myregistry/mnist-cpu:latest
```

