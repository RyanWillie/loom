# Dev Container Configuration

This directory contains the VS Code Dev Container configuration for the MNIST Neural Network project.

## What is a Dev Container?

A Dev Container is a fully-featured development environment running inside a Docker container. It provides:

- Consistent development environment across different machines
- All dependencies pre-installed
- Preconfigured VS Code settings and extensions
- Isolated from your host system

## Usage

### Prerequisites

1. Install [Docker](https://docs.docker.com/get-docker/)
2. Install [VS Code](https://code.visualstudio.com/)
3. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Opening in Dev Container

1. Open this project folder in VS Code
2. VS Code will detect the `.devcontainer` configuration
3. Click "Reopen in Container" when prompted
   - Or use Command Palette (F1) → "Dev Containers: Reopen in Container"
4. Wait for the container to build and start
5. You're ready to develop!

## Features

The Dev Container includes:

### Build Tools
- CMake 3.x
- Ninja build system
- Make (fallback)
- GCC and Clang compilers

### Development Tools
- clang-format (code formatting)
- clang-tidy (static analysis)
- GDB (debugger)
- Valgrind (memory profiler)

### Sanitizers
- AddressSanitizer (ASan)
- UndefinedBehaviorSanitizer (UBSan)
- ThreadSanitizer (TSan)

### VS Code Extensions
- C/C++ Extension Pack
- CMake Tools
- clangd
- clang-format
- clang-tidy

## Building Inside Dev Container

Once inside the Dev Container, use the build script:

```bash
# Debug build
./build.sh debug

# Release build
./build.sh release

# With sanitizers
./build.sh asan

# Run tests
./build.sh test
```

## Debugging

The Dev Container is configured for debugging:

1. Set breakpoints in your code
2. Press `F5` or go to "Run and Debug"
3. Select a debug configuration
4. Start debugging!

## GPU Support

**Note:** This Dev Container uses the CPU-only configuration. For GPU/CUDA development:

1. Use the Docker GPU image directly:
   ```bash
   docker build -t mnist-gpu -f docker/Dockerfile.gpu .
   docker run --gpus all -it mnist-gpu
   ```

2. Or modify `.devcontainer/devcontainer.json` to use `Dockerfile.gpu` and add:
   ```json
   "runArgs": ["--gpus", "all"]
   ```

## Customization

Edit `.devcontainer/devcontainer.json` to:
- Add VS Code extensions
- Change settings
- Install additional packages
- Modify environment variables

After changes, rebuild the container:
- Command Palette (F1) → "Dev Containers: Rebuild Container"

## Troubleshooting

### Container won't build
- Check Docker is running: `docker ps`
- Try rebuilding: F1 → "Dev Containers: Rebuild Container Without Cache"

### Permission issues
- The container runs as root by default
- Files created in the container are owned by root

### GDB not working
- The container has `--cap-add=SYS_PTRACE` for debugging support
- If still issues, check Docker security settings

