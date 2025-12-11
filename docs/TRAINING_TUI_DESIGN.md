# Training Infrastructure & TUI Design Document

**Version:** 1.0  
**Date:** December 2025  
**Status:** Design Phase

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Design Principles](#design-principles)
4. [Component Specifications](#component-specifications)
5. [Profiler Integration](#profiler-integration)
6. [TUI Views & Layouts](#tui-views--layouts)
7. [Implementation Plan](#implementation-plan)
8. [File Structure](#file-structure)
9. [API Reference](#api-reference)
10. [Examples](#examples)
11. [Testing Strategy](#testing-strategy)

---

## Overview

### Purpose

Create a flexible training infrastructure with an interactive Terminal User Interface (TUI) for real-time monitoring and control of model training across any architecture.

### Key Features

- **Architecture-agnostic**: Works with any `nn::Module` subclass
- **Callback-based**: User provides training logic (loss, optimizer, metrics)
- **Interactive TUI**: Real-time visualization with pause/save/quit controls
- **Profiler integration**: First-class support for timing and memory profiling
- **Multi-view interface**: Separate views for training metrics and system profiling
- **Flexible monitoring**: Pluggable monitor system (TUI, console, tensorboard, etc.)
- **Checkpoint support**: User-defined save/load with automatic integration

### Non-Goals (Out of Scope)

- Loss function implementations (user-provided)
- Optimizer implementations (user-provided)
- Learning rate schedulers (user-provided)
- Model architecture definitions (user-provided)

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│  (Model, Loss, Optimizer, Metrics, Profiler, DataLoader)        │
└────────────────────────────┬────────────────────────────────────┘
                             │ Callbacks
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Trainer                                  │
│  - Orchestrates training loop                                    │
│  - Manages state (epoch, batch, pause/stop flags)               │
│  - Collects metrics from user callbacks                         │
│  - Integrates profiler data                                      │
│  - Notifies monitors                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │ State updates
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TrainingMonitor (Interface)                   │
│  - onBatchComplete(), onEpochEnd(), etc.                        │
│  - shouldPause(), shouldStop(), shouldSave()                    │
└────────┬────────────────────────────────────────────────────────┘
         │
         ├──► TUIMonitor (Interactive TUI)
         │     - Multi-view interface
         │     - Keyboard input handling
         │     - Real-time charts and graphs
         │
         ├──► ConsoleMonitor (Simple text output)
         │     - Uses Logger class
         │     - Non-interactive
         │
         └──► [Future: TensorboardMonitor, WandBMonitor, etc.]
```

### Data Flow

```
Training Loop Iteration:
    1. User forward() → predictions
    2. User loss() → loss value
    3. User backward() → gradients computed
    4. User optimizer.step() → weights updated
    5. User metrics() → {"accuracy": 0.89, ...}
    6. Profiler (optional) → timing & memory data
    7. Trainer packages everything → TrainingState
    8. Trainer notifies monitors → TUI updates
    9. TUI checks user input → pause/save/quit flags
    10. Trainer responds to flags → pause/save/quit
```

---

## Design Principles

### 1. Separation of Concerns

- **Trainer**: Orchestration only (no training logic)
- **Monitor**: Display & interaction only (no training logic)
- **User Code**: All domain logic (forward, backward, loss, optimizer)

### 2. Flexibility

- Pluggable monitors (can have multiple simultaneously)
- User-defined metrics (arbitrary key-value pairs)
- Optional profiler integration
- Custom checkpoint logic

### 3. Type Safety

- Strong typing for callbacks
- Const-correctness
- No raw pointers for ownership

### 4. Thread Safety

- Atomic flags for control (pause/stop/save)
- Mutex-protected shared state
- Lock-free queues where appropriate

### 5. Zero-Cost Abstractions

- No monitoring overhead if not enabled
- Virtual dispatch only for monitors
- Minimal allocations in hot path

---

## Component Specifications

### 1. Metrics System

#### Purpose
Generic container for training metrics with historical tracking.

#### Files
- `include/loom/training/metrics.h`
- `src/common/training/metrics.cpp`

#### Key Types

```cpp
struct MetricsSnapshot {
    // Training state
    size_t epoch;
    size_t total_epochs;
    size_t batch;
    size_t total_batches;
    
    // User-provided metrics (flexible)
    std::unordered_map<std::string, float> train_metrics;  // e.g., {"loss": 0.34, "accuracy": 0.89}
    std::unordered_map<std::string, float> val_metrics;
    
    // Timing
    double batch_time_ms;
    double epoch_time_ms;
    
    // System
    double samples_per_sec;
};

class TrainingHistory {
public:
    void record(const MetricsSnapshot& snapshot);
    
    // Query historical data
    std::vector<float> getHistory(const std::string& metric_name, bool validation = false) const;
    const MetricsSnapshot& getLatest() const;
    const MetricsSnapshot& getBest(const std::string& metric_name, bool minimize = true) const;
    
    // Export
    void exportToJSON(const std::string& path) const;
    void exportToCSV(const std::string& path) const;
    
    // Statistics
    float getAverage(const std::string& metric_name, size_t last_n_epochs = 0) const;
    float getStdDev(const std::string& metric_name, size_t last_n_epochs = 0) const;
    
private:
    std::vector<MetricsSnapshot> history_;
};
```

#### Design Notes

- **Flexible metrics**: Use string keys for arbitrary metrics
- **Efficient storage**: Vector-based for cache-friendly iteration
- **Export support**: JSON for tools, CSV for spreadsheets
- **Statistical queries**: Common operations built-in

---

### 2. Profiler Integration

#### Purpose
Bridge between user's Profiler implementation and Trainer/TUI.

#### Files
- `include/loom/training/profiler_bridge.h`

#### Key Types

```cpp
struct ProfileSnapshot {
    // Operation timing breakdown
    struct OperationTiming {
        std::string name;           // e.g., "forward_pass", "backward_pass"
        double duration_ms;
        double percentage;          // of total batch time
        size_t call_count;          // number of times called in this batch
    };
    std::vector<OperationTiming> operations;
    
    // Memory usage
    struct MemoryUsage {
        size_t model_params_bytes;
        size_t gradients_bytes;
        size_t activations_bytes;
        size_t optimizer_state_bytes;
        size_t peak_usage_bytes;
        size_t total_available_bytes;
    };
    MemoryUsage memory;
    
    // Performance insights
    std::vector<std::string> warnings;      // e.g., "Backward pass is bottleneck"
    std::vector<std::string> suggestions;   // e.g., "Consider increasing batch size"
    
    // Aggregate statistics
    double total_batch_time_ms;
    double throughput_samples_per_sec;
    double gpu_utilization_percent;  // if applicable
};

// Interface that user's Profiler implements
class IProfiler {
public:
    virtual ~IProfiler() = default;
    
    // Get current profiling data
    virtual ProfileSnapshot getSnapshot() const = 0;
    
    // Reset profiling counters (called at epoch start)
    virtual void reset() = 0;
    
    // Optional: Enable/disable profiling (performance impact)
    virtual void setEnabled(bool enabled) { /* optional */ }
};
```

#### User Implementation Example

```cpp
class MyProfiler : public IProfiler {
    // User implements timing/memory tracking
    ProfileSnapshot getSnapshot() const override {
        ProfileSnapshot snapshot;
        snapshot.operations = {
            {"forward_pass", forward_time_ms_, forward_time_ms_ / total_time_ * 100},
            {"backward_pass", backward_time_ms_, backward_time_ms_ / total_time_ * 100},
            // ...
        };
        snapshot.memory = getMemoryStats();
        return snapshot;
    }
    
    void reset() override {
        forward_time_ms_ = 0;
        backward_time_ms_ = 0;
        // ...
    }
};
```

#### Design Notes

- **Interface-based**: User implements `IProfiler`, full control over implementation
- **Rich data**: Timing, memory, insights all in one snapshot
- **Optional**: Profiler can be nullptr, no overhead if not used
- **Extensible**: User can add custom fields to their Profiler class

---

### 3. Training Monitor Interface

#### Purpose
Abstract interface for monitoring training progress (TUI, logging, tensorboard, etc.).

#### Files
- `include/loom/training/monitor.h`

#### Key Types

```cpp
struct TrainingState {
    MetricsSnapshot metrics;
    ProfileSnapshot profile;         // Empty if no profiler attached
    std::string status_message;      // Optional status updates
};

class TrainingMonitor {
public:
    virtual ~TrainingMonitor() = default;
    
    // Lifecycle callbacks
    virtual void onTrainStart(size_t total_epochs) = 0;
    virtual void onEpochStart(size_t epoch) = 0;
    virtual void onBatchComplete(const TrainingState& state) = 0;
    virtual void onEpochEnd(const TrainingState& state) = 0;
    virtual void onTrainEnd(const TrainingHistory& history) = 0;
    
    // User input (interactive monitors only)
    virtual bool shouldPause() = 0;
    virtual bool shouldStop() = 0;
    virtual bool shouldSave() = 0;
    
    // Event notifications
    virtual void onCheckpointSaved(const std::string& path) = 0;
    virtual void onError(const std::string& error) = 0;
};
```

#### Design Notes

- **Complete state**: All relevant info in `TrainingState`
- **Non-intrusive**: Read-only access to training state
- **Control flow**: Monitors suggest actions via `should*()` methods
- **Extensible**: Easy to add new monitor types

---

### 4. Trainer Class

#### Purpose
Orchestrates the training loop, manages state, integrates monitors.

#### Files
- `include/loom/training/trainer.h`
- `src/common/training/trainer.cpp`

#### Key Types

```cpp
// Callback function signatures
using ForwardFunction = std::function<Tensor(const Tensor&)>;
using LossFunction = std::function<Tensor(const Tensor&, const Tensor&)>;
using BackwardFunction = std::function<void(const Tensor&)>;
using OptimizerStepFunction = std::function<void()>;
using MetricsFunction = std::function<std::unordered_map<std::string, float>(const Tensor&, const Tensor&)>;
using SaveFunction = std::function<void(const std::string&, size_t epoch)>;
using LoadFunction = std::function<size_t(const std::string&)>;  // Returns start epoch

class Trainer {
public:
    Trainer();
    ~Trainer();
    
    // ========================================================================
    // Configuration (Fluent API)
    // ========================================================================
    
    // Data loaders
    Trainer& withTrainLoader(DataLoader& loader);
    Trainer& withValLoader(DataLoader& loader);
    
    // Training logic (user-provided)
    Trainer& withForward(ForwardFunction fn);
    Trainer& withLoss(LossFunction fn);
    Trainer& withBackward(BackwardFunction fn);
    Trainer& withOptimizerStep(OptimizerStepFunction fn);
    
    // Metrics computation (user-provided)
    Trainer& withMetrics(MetricsFunction train_fn, MetricsFunction val_fn = nullptr);
    
    // Monitoring
    Trainer& withMonitor(std::unique_ptr<TrainingMonitor> monitor);
    Trainer& withTUI(bool show_profiler = true);        // Convenience: adds TUIMonitor
    Trainer& withConsoleLogging();                      // Convenience: adds ConsoleMonitor
    
    // Profiler integration (optional)
    Trainer& withProfiler(IProfiler* profiler);
    
    // Checkpointing (user-provided save/load logic)
    Trainer& withCheckpoint(SaveFunction save_fn, LoadFunction load_fn, const std::string& path);
    
    // Options
    Trainer& withEpochValidation(bool enabled = true);  // Validate after each epoch
    Trainer& withAutoSave(size_t every_n_epochs);       // Auto-save checkpoints
    
    // ========================================================================
    // Training Execution
    // ========================================================================
    
    void train(size_t num_epochs);
    void trainFromCheckpoint(const std::string& path);
    
    // Manual control (can be called from another thread)
    void pause();
    void resume();
    void stop();
    void saveCheckpoint();
    
    // ========================================================================
    // Access
    // ========================================================================
    
    const TrainingHistory& history() const;
    bool isPaused() const;
    bool isStopped() const;
    size_t currentEpoch() const;
    
private:
    // Data
    DataLoader* train_loader_{nullptr};
    DataLoader* val_loader_{nullptr};
    
    // User-provided callbacks
    ForwardFunction forward_fn_;
    LossFunction loss_fn_;
    BackwardFunction backward_fn_;
    OptimizerStepFunction optimizer_step_fn_;
    MetricsFunction train_metrics_fn_;
    MetricsFunction val_metrics_fn_;
    SaveFunction save_fn_;
    LoadFunction load_fn_;
    
    // Profiler (optional)
    IProfiler* profiler_{nullptr};
    
    // Monitoring
    std::vector<std::unique_ptr<TrainingMonitor>> monitors_;
    TrainingHistory history_;
    
    // State management
    std::atomic<bool> paused_{false};
    std::atomic<bool> stopped_{false};
    std::atomic<bool> save_requested_{false};
    std::string checkpoint_path_;
    size_t current_epoch_{0};
    size_t start_epoch_{0};
    
    // Options
    bool validate_each_epoch_{true};
    size_t auto_save_interval_{0};  // 0 = disabled
    
    // Timing
    std::chrono::steady_clock::time_point batch_start_;
    std::chrono::steady_clock::time_point epoch_start_;
    std::chrono::steady_clock::time_point train_start_;
    
    // Performance tracking
    size_t total_samples_processed_{0};
    
    // ========================================================================
    // Private Implementation
    // ========================================================================
    
    void trainEpoch(size_t epoch);
    void validateEpoch(size_t epoch);
    
    TrainingState createState(const MetricsSnapshot& metrics) const;
    void notifyMonitors(const std::function<void(TrainingMonitor&)>& fn);
    void checkUserInput();
    
    void handlePause();
    void handleSave();
    
    double computeThroughput() const;
    
    // Validation helpers
    void validateConfiguration() const;  // Check all required callbacks are set
};
```

#### Design Notes

- **Fluent API**: Method chaining for readable configuration
- **Validation**: Checks required callbacks before training starts
- **Thread-safe control**: Atomic flags for pause/stop/save
- **Multiple monitors**: Can attach TUI + tensorboard + custom simultaneously
- **Flexible timing**: Tracks batch/epoch/total time automatically

---

### 5. TUI Monitor

#### Purpose
Interactive terminal UI with multi-view support for training metrics and profiler data.

#### Files
- `include/loom/training/tui_monitor.h`
- `src/common/training/tui_monitor.cpp`

#### Key Types

```cpp
enum class TUIView {
    TRAINING,  // Loss, accuracy, progress bars
    PROFILER,  // Timing breakdown, memory usage
    HISTORY    // Historical charts over epochs
};

class TUIMonitor : public TrainingMonitor {
public:
    explicit TUIMonitor(bool enable_profiler_view = true);
    ~TUIMonitor();
    
    // TrainingMonitor interface
    void onTrainStart(size_t total_epochs) override;
    void onEpochStart(size_t epoch) override;
    void onBatchComplete(const TrainingState& state) override;
    void onEpochEnd(const TrainingState& state) override;
    void onTrainEnd(const TrainingHistory& history) override;
    
    bool shouldPause() override;
    bool shouldStop() override;
    bool shouldSave() override;
    
    void onCheckpointSaved(const std::string& path) override;
    void onError(const std::string& error) override;
    
private:
    // UI thread (runs independently)
    std::thread ui_thread_;
    std::atomic<bool> running_{true};
    
    // Shared state (protected by mutex)
    std::mutex state_mutex_;
    TrainingState current_state_;
    std::vector<float> train_loss_history_;
    std::vector<float> val_loss_history_;
    std::deque<std::string> log_messages_;  // Recent messages
    
    // User input state
    std::atomic<bool> pause_requested_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> save_requested_{false};
    
    // View state
    TUIView current_view_{TUIView::TRAINING};
    bool profiler_enabled_;
    int scroll_offset_{0};
    
    // Configuration
    size_t max_loss_history_{100};  // Keep last N epochs
    size_t max_log_messages_{20};
    
    // ========================================================================
    // UI Thread
    // ========================================================================
    
    void renderLoop();
    ftxui::Component createUI();
    void handleInput(ftxui::Event event);
    
    // ========================================================================
    // View Renderers
    // ========================================================================
    
    ftxui::Element renderView();
    
    // Training View
    ftxui::Element renderTrainingView();
    ftxui::Element renderHeader();
    ftxui::Element renderProgressBars();
    ftxui::Element renderMetricsTable();
    ftxui::Element renderLossChart();
    ftxui::Element renderLogMessages();
    
    // Profiler View
    ftxui::Element renderProfilerView();
    ftxui::Element renderTimingBreakdown();
    ftxui::Element renderMemoryUsage();
    ftxui::Element renderBottlenecksPanel();
    
    // History View
    ftxui::Element renderHistoryView();
    ftxui::Element renderFullLossChart();
    ftxui::Element renderMetricsHistory();
    
    // Common
    ftxui::Element renderControls();
    ftxui::Element renderStatusBar();
    
    // ========================================================================
    // Helpers
    // ========================================================================
    
    std::string formatDuration(double seconds) const;
    std::string formatMemory(size_t bytes) const;
    std::string formatPercentage(float value) const;
    ftxui::Color getMetricColor(float current, float previous) const;
    
    void updateLossHistory(const TrainingState& state);
    void addLogMessage(const std::string& message, LogLevel level = LogLevel::INFO);
};
```

#### UI Layout (Training View)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LOOM Training Monitor                     [TAB] Views: ●Train  Profiler │
├─────────────────────────────────────────────────────────────────────────┤
│ Model: MLP (784→128→10)                              Device: CPU        │
│ Epoch: 15/100    Batch: 543/938         Elapsed: 02:34:15   ETA: 14:23 │
├─────────────────────────────────────────────────────────────────────────┤
│ Progress                                                                 │
│   Epoch:  [████████████████░░░░░░░░] 15%                               │
│   Batch:  [██████████████████████░░] 57.8%                             │
├─────────────────────────────────────────────────────────────────────────┤
│ Metrics                          Current      Best      Change          │
│   Train Loss                     0.3421      0.2134    ↓ 12.3%         │
│   Train Accuracy                 89.2%       91.5%     ↑ 2.1%          │
│   Val Loss                       0.4123      0.2876    ↓ 8.7%          │
│   Val Accuracy                   87.4%       89.8%     ↑ 1.8%          │
│   Throughput                     1247 samples/sec                       │
├─────────────────────────────────────────────────────────────────────────┤
│ Loss History (last 50 epochs)                                           │
│  1.2│                                                                    │
│  0.9│╮                                                                   │
│  0.6│ ╰╮                                                                 │
│  0.3│   ╰─╮─────────────────────                                        │
│  0.0│      ╰────────────────────────────────                            │
│     └────────────────────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────────────────────┤
│ Recent Activity                                                          │
│   [02:34:12] Checkpoint saved: checkpoints/mnist_epoch_14.loom         │
│   [02:31:45] Validation complete - Acc: 87.4%                          │
│   [02:28:33] Epoch 14/100 complete - Loss: 0.3421                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Controls: [P]ause [S]ave [Q]uit [TAB]switch view [↑/↓]scroll          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### UI Layout (Profiler View)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LOOM Training Monitor                     [TAB] Views:  Train ●Profiler │
├─────────────────────────────────────────────────────────────────────────┤
│ Operation Timing (per batch)                     Epoch: 15/100          │
├─────────────────────────────────────────────────────────────────────────┤
│ Operation          Time        Percentage                               │
│ ───────────────────────────────────────────────────────────────────     │
│ Forward Pass       12.3ms      ████████████░░░░░░░ 48%                 │
│ Backward Pass      15.7ms      ███████████████░░░░ 61%                 │
│ Optimizer Step     3.2ms       ███░░░░░░░░░░░░░░░░ 12%                 │
│ Data Loading       2.1ms       ██░░░░░░░░░░░░░░░░░  8%                 │
│ Other              0.8ms       █░░░░░░░░░░░░░░░░░░  3%                 │
│ ───────────────────────────────────────────────────────────────────     │
│ Total Batch Time   34.1ms                                               │
│ Throughput         1876 samples/sec                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Memory Usage                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│ Component          Size        Usage                                    │
│ ───────────────────────────────────────────────────────────────────     │
│ Model Parameters   42.3 MB                                              │
│ Gradients          42.3 MB                                              │
│ Activations        128.7 MB                                             │
│ Optimizer State    84.6 MB                                              │
│ ───────────────────────────────────────────────────────────────────     │
│ Total Allocated    297.9 MB                                             │
│ Peak Usage         312.4 MB    ███████████░░░░░░░ 19% of available     │
│ Available          1536.0 MB                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Performance Insights                                                     │
│   ⚠ Backward pass is taking 61% of batch time (bottleneck)            │
│   ✓ Data loading is efficient (< 10%)                                  │
│   ✓ Memory usage is healthy (< 50% of available)                       │
│   💡 Consider: Mixed precision training to reduce memory                │
├─────────────────────────────────────────────────────────────────────────┤
│ Controls: [P]ause [S]ave [Q]uit [TAB]switch view                       │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Keyboard Controls

| Key | Action |
|-----|--------|
| `TAB` | Switch between views (Training → Profiler → History → Training) |
| `P` | Pause/Resume training |
| `S` | Save checkpoint immediately |
| `Q` | Quit training (with confirmation) |
| `↑/↓` | Scroll log messages / history |
| `ESC` | Cancel confirmation dialogs |

#### Design Notes

- **Separate thread**: UI runs independently, doesn't block training
- **Lock-free reads**: TUI reads state with minimal locking
- **Adaptive refresh**: Updates only when state changes (battery-friendly)
- **Unicode graphics**: Beautiful charts using box-drawing characters
- **Color coding**: Red/green for metrics getting worse/better
- **Responsive**: Works on different terminal sizes (minimum 80x24)

---

### 6. Console Monitor

#### Purpose
Simple non-interactive logging fallback for environments without TUI support.

#### Files
- `include/loom/training/console_monitor.h`
- `src/common/training/console_monitor.cpp`

#### Implementation

```cpp
class ConsoleMonitor : public TrainingMonitor {
public:
    ConsoleMonitor();
    
    void onTrainStart(size_t total_epochs) override;
    void onEpochStart(size_t epoch) override;
    void onBatchComplete(const TrainingState& state) override;
    void onEpochEnd(const TrainingState& state) override;
    void onTrainEnd(const TrainingHistory& history) override;
    
    // Non-interactive: always return false
    bool shouldPause() override { return false; }
    bool shouldStop() override { return false; }
    bool shouldSave() override { return false; }
    
    void onCheckpointSaved(const std::string& path) override;
    void onError(const std::string& error) override;
    
private:
    Logger& logger_;
    size_t update_frequency_{10};  // Log every N batches
};
```

#### Output Example

```
[2025-12-07 14:23:45] [TRAIN] === Training Started: 100 epochs ===
[2025-12-07 14:23:45] [TRAIN] Epoch 1/100
[2025-12-07 14:23:47] [TRAIN]   Batch 100/938 - Loss: 0.4521 - Acc: 85.3% - 1234 samples/sec
[2025-12-07 14:23:49] [TRAIN]   Batch 200/938 - Loss: 0.4012 - Acc: 86.7% - 1245 samples/sec
...
[2025-12-07 14:24:15] [TRAIN] Epoch 1/100 complete - Train Loss: 0.3421 - Train Acc: 89.2%
[2025-12-07 14:24:18] [TRAIN] Validation - Val Loss: 0.4123 - Val Acc: 87.4%
```

---

## Profiler Integration

### Why First-Class Integration?

Profiler data is **system observability**, distinct from training metrics:

- **Training metrics** (loss, accuracy): Model performance
- **Profiler data** (timing, memory): System performance

These serve different purposes and should be displayed separately.

### Integration Points

1. **Trainer Configuration**: `trainer.withProfiler(&profiler)`
2. **Data Collection**: Trainer queries profiler after each batch
3. **Monitor Distribution**: Profiler data included in `TrainingState`
4. **TUI Display**: Dedicated profiler view

### User Workflow

```cpp
// 1. User implements IProfiler
class MyProfiler : public IProfiler {
    ProfileSnapshot getSnapshot() const override { /* ... */ }
    void reset() override { /* ... */ }
};

// 2. User creates profiler instance
MyProfiler profiler;

// 3. User attaches to trainer
trainer.withProfiler(&profiler);

// 4. Trainer automatically:
//    - Calls profiler.getSnapshot() after each batch
//    - Passes data to monitors
//    - TUI displays in profiler view
```

### Optional Nature

- If no profiler attached: `TrainingState::profile` is empty
- TUI adapts: Profiler view disabled or shows "No profiler attached"
- Zero overhead: No profiler calls if not attached

---

## TUI Views & Layouts

### View 1: Training (Default)

**Purpose**: Real-time training progress and metrics

**Components**:
- Header (model info, device, elapsed time, ETA)
- Progress bars (epoch and batch)
- Metrics table (loss, accuracy, custom metrics)
- Loss chart (last 50-100 epochs)
- Recent activity log
- Control hints

**Update Frequency**: Every batch (throttled to ~10 FPS)

### View 2: Profiler

**Purpose**: System performance analysis

**Components**:
- Operation timing breakdown (bar chart)
- Memory usage breakdown
- Performance insights (warnings/suggestions)
- Bottleneck analysis
- GPU utilization (if applicable)

**Update Frequency**: Every batch or every N batches

**Availability**: Only if profiler attached

### View 3: History

**Purpose**: Long-term trends over entire training

**Components**:
- Full loss/accuracy charts (all epochs)
- Metric statistics (mean, std dev, trend)
- Best epoch information
- Training summary

**Update Frequency**: On-demand (when view is active)

### Responsive Design

- **Minimum size**: 80 columns × 24 rows
- **Recommended**: 100 columns × 30 rows
- **Adaptive**: Adjusts components based on terminal size
- **Scrolling**: For content that doesn't fit

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### Tasks

1. **Metrics System**
   - Implement `MetricsSnapshot` struct
   - Implement `TrainingHistory` class
   - Add JSON/CSV export
   - Unit tests

2. **Profiler Bridge**
   - Define `ProfileSnapshot` struct
   - Define `IProfiler` interface
   - Create mock profiler for testing
   - Unit tests

3. **Monitor Interface**
   - Define `TrainingState` struct
   - Define `TrainingMonitor` interface
   - Documentation

4. **Console Monitor**
   - Implement basic ConsoleMonitor
   - Integration with existing Logger
   - Test with dummy training loop

#### Deliverables

- All header files created
- Basic implementations complete
- Unit tests passing
- Can run simple training with console output

---

### Phase 2: Trainer Core (Week 1-2)

#### Tasks

1. **Trainer Class Structure**
   - Implement configuration methods (fluent API)
   - Implement validation logic
   - Add state management (pause/stop/save)

2. **Training Loop**
   - Implement `train()` method
   - Implement `trainEpoch()` and `validateEpoch()`
   - Add timing tracking
   - Add throughput calculation

3. **Monitor Integration**
   - Implement monitor notification system
   - Add multi-monitor support
   - Implement user input checking

4. **Checkpoint Integration**
   - Implement checkpoint save/load hooks
   - Add auto-save functionality
   - Test checkpoint resume

#### Deliverables

- Functional Trainer class
- Can train MNIST with console monitoring
- Pause/resume/save functionality works
- Unit and integration tests passing

---

### Phase 3: TUI Implementation (Week 2)

#### Tasks

1. **FTXUI Setup**
   - Add FTXUI dependency to CMake
   - Test FTXUI installation
   - Create minimal TUI example

2. **TUI Monitor Structure**
   - Implement threading model
   - Add shared state management
   - Implement input handling

3. **Training View**
   - Implement header
   - Implement progress bars
   - Implement metrics table
   - Implement loss chart
   - Implement log messages

4. **View Switching**
   - Implement TAB key handling
   - Add view state management
   - Test view transitions

5. **User Controls**
   - Implement pause/resume (P key)
   - Implement save (S key)
   - Implement quit (Q key with confirmation)

#### Deliverables

- Functional TUI with Training view
- Interactive controls working
- Can train MNIST with TUI
- View updates in real-time

---

### Phase 4: Profiler View (Week 2-3)

#### Tasks

1. **Profiler View Layout**
   - Implement timing breakdown display
   - Implement memory usage display
   - Implement insights panel

2. **Mock Profiler**
   - Create realistic mock profiler for testing
   - Generate synthetic profiling data
   - Test TUI with mock data

3. **Integration**
   - Test with trainer + mock profiler
   - Verify data flow
   - Test view switching

#### Deliverables

- Functional Profiler view in TUI
- Can display profiling data
- Ready for real profiler integration

---

### Phase 5: Polish & Examples (Week 3)

#### Tasks

1. **History View**
   - Implement full historical charts
   - Add statistics display
   - Add best epoch info

2. **MNIST Example**
   - Create complete training example
   - Add profiler example
   - Add checkpoint save/load example

3. **Documentation**
   - API reference
   - User guide
   - Example walkthrough

4. **Testing**
   - Integration tests
   - Edge case testing
   - Performance testing

#### Deliverables

- Complete feature set
- Full MNIST example
- Documentation complete
- All tests passing

---

### Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 3-4 days | Core infrastructure working |
| Phase 2 | 3-4 days | Trainer functional with console |
| Phase 3 | 4-5 days | TUI with Training view |
| Phase 4 | 2-3 days | TUI with Profiler view |
| Phase 5 | 3-4 days | Polish, examples, docs |
| **Total** | **3 weeks** | **Complete feature** |

---

## File Structure

```
loom/
├── CMakeLists.txt                       [MODIFY - Add FTXUI]
│
├── include/loom/training/               [NEW DIRECTORY]
│   ├── metrics.h                        [NEW - Metrics types and history]
│   ├── profiler_bridge.h                [NEW - Profiler integration interface]
│   ├── monitor.h                        [NEW - Monitor interface]
│   ├── trainer.h                        [NEW - Main Trainer class]
│   ├── tui_monitor.h                    [NEW - TUI implementation]
│   └── console_monitor.h                [NEW - Console implementation]
│
├── src/common/training/                 [NEW DIRECTORY]
│   ├── CMakeLists.txt                   [NEW - Build config]
│   ├── metrics.cpp                      [NEW - Metrics implementation]
│   ├── trainer.cpp                      [NEW - Trainer implementation]
│   ├── tui_monitor.cpp                  [NEW - TUI implementation]
│   └── console_monitor.cpp              [NEW - Console implementation]
│
├── examples/mnist/
│   ├── CMakeLists.txt                   [MODIFY - Add new examples]
│   ├── basic.cpp                        [RENAME from main.cpp]
│   └── train_with_tui.cpp               [NEW - Full training example]
│
├── tests/
│   ├── CMakeLists.txt                   [MODIFY]
│   ├── test_metrics.cpp                 [NEW]
│   ├── test_history.cpp                 [NEW]
│   ├── test_trainer.cpp                 [NEW]
│   └── test_tui_mock.cpp                [NEW - Test with mock profiler]
│
└── docs/
    ├── TRAINING_TUI_DESIGN.md           [THIS FILE]
    └── training_user_guide.md           [NEW - User documentation]
```

---

## API Reference

### Quick Start

```cpp
#include "loom/training/trainer.h"
#include "loom/training/tui_monitor.h"

// Setup model and data
auto model = std::make_shared<MyModel>();
DataLoader train_loader(...);
DataLoader val_loader(...);

// Configure trainer
training::Trainer trainer;
trainer.withTrainLoader(train_loader)
       .withValLoader(val_loader)
       .withForward([&](const Tensor& x) { return model->forward(x); })
       .withLoss([&](const Tensor& pred, const Tensor& target) { 
           return loss_fn->compute(pred, target); 
       })
       .withBackward([&](const Tensor& loss) { loss.backward(); })
       .withOptimizerStep([&]() { 
           optimizer->step(); 
           model->zeroGrad(); 
       })
       .withTUI();

// Train
trainer.train(100);
```

### Configuration Methods

#### Data

```cpp
Trainer& withTrainLoader(DataLoader& loader);
Trainer& withValLoader(DataLoader& loader);
```

#### Training Logic

```cpp
Trainer& withForward(ForwardFunction fn);
Trainer& withLoss(LossFunction fn);
Trainer& withBackward(BackwardFunction fn);
Trainer& withOptimizerStep(OptimizerStepFunction fn);
```

#### Metrics

```cpp
// Compute training metrics
Trainer& withMetrics(
    MetricsFunction train_fn,
    MetricsFunction val_fn = nullptr
);

// Example:
trainer.withMetrics(
    [](const Tensor& pred, const Tensor& target) {
        return std::unordered_map<std::string, float>{
            {"accuracy", computeAccuracy(pred, target)},
            {"f1", computeF1(pred, target)}
        };
    }
);
```

#### Monitoring

```cpp
// Add custom monitor
Trainer& withMonitor(std::unique_ptr<TrainingMonitor> monitor);

// Convenience methods
Trainer& withTUI(bool show_profiler = true);
Trainer& withConsoleLogging();

// Multiple monitors
trainer.withTUI()
       .withConsoleLogging()  // Both TUI and console logs
       .withMonitor(std::make_unique<MyCustomMonitor>());
```

#### Profiler

```cpp
// Attach profiler (user-implemented)
Trainer& withProfiler(IProfiler* profiler);

// Example:
MyProfiler profiler;
trainer.withProfiler(&profiler);
```

#### Checkpoints

```cpp
Trainer& withCheckpoint(
    SaveFunction save_fn,
    LoadFunction load_fn,
    const std::string& path
);

// Example:
trainer.withCheckpoint(
    // Save
    [&](const std::string& path, size_t epoch) {
        saveModelAndOptimizer(path, model, optimizer, epoch);
    },
    // Load
    [&](const std::string& path) {
        return loadModelAndOptimizer(path, model, optimizer);
    },
    "checkpoints/model"
);

// Auto-save every N epochs
trainer.withAutoSave(5);  // Save every 5 epochs
```

#### Options

```cpp
Trainer& withEpochValidation(bool enabled = true);
Trainer& withAutoSave(size_t every_n_epochs);
```

### Training Execution

```cpp
// Train from scratch
void train(size_t num_epochs);

// Resume from checkpoint
void trainFromCheckpoint(const std::string& path);

// Manual control
void pause();
void resume();
void stop();
void saveCheckpoint();
```

### Access

```cpp
const TrainingHistory& history() const;
bool isPaused() const;
bool isStopped() const;
size_t currentEpoch() const;
```

---

## Examples

### Example 1: Minimal Setup

```cpp
#include "loom/training/trainer.h"

training::Trainer trainer;
trainer.withTrainLoader(train_loader)
       .withForward([&](const Tensor& x) { return model->forward(x); })
       .withLoss([&](const Tensor& pred, const Tensor& target) { 
           return loss_fn->compute(pred, target); 
       })
       .withBackward([&](const Tensor& loss) { loss.backward(); })
       .withOptimizerStep([&]() { 
           optimizer->step(); 
           model->zeroGrad(); 
       })
       .withConsoleLogging();  // Simple console output

trainer.train(50);
```

### Example 2: Full Featured

```cpp
#include "loom/training/trainer.h"
#include "loom/profiler.h"

// Your profiler implementation
Profiler profiler;

training::Trainer trainer;
trainer.withTrainLoader(train_loader)
       .withValLoader(val_loader)
       .withForward([&](const Tensor& x) { return model->forward(x); })
       .withLoss([&](const Tensor& pred, const Tensor& target) { 
           return cross_entropy(pred, target); 
       })
       .withBackward([&](const Tensor& loss) { loss.backward(); })
       .withOptimizerStep([&]() { 
           optimizer->step(); 
           model->zeroGrad(); 
       })
       .withMetrics(
           // Training metrics
           [](const Tensor& pred, const Tensor& target) {
               return std::unordered_map<std::string, float>{
                   {"accuracy", computeAccuracy(pred, target)},
                   {"top5", computeTop5Accuracy(pred, target)}
               };
           },
           // Validation metrics
           [](const Tensor& pred, const Tensor& target) {
               return std::unordered_map<std::string, float>{
                   {"accuracy", computeAccuracy(pred, target)},
                   {"top5", computeTop5Accuracy(pred, target)}
               };
           }
       )
       .withProfiler(&profiler)
       .withCheckpoint(
           [&](const std::string& path, size_t epoch) {
               saveCheckpoint(path, model, optimizer, epoch);
           },
           [&](const std::string& path) {
               return loadCheckpoint(path, model, optimizer);
           },
           "checkpoints/model"
       )
       .withAutoSave(5)
       .withTUI(true);

trainer.train(100);

// Export training history
trainer.history().exportToJSON("training_history.json");
```

### Example 3: Custom Monitor

```cpp
class MyCustomMonitor : public TrainingMonitor {
public:
    void onEpochEnd(const TrainingState& state) override {
        // Send to your custom logging service
        sendToLoggingService(state.metrics);
    }
    
    // Implement other methods...
};

trainer.withTUI()
       .withMonitor(std::make_unique<MyCustomMonitor>());
```

### Example 4: Resuming from Checkpoint

```cpp
training::Trainer trainer;
// ... configure trainer ...

// Resume training from checkpoint
trainer.trainFromCheckpoint("checkpoints/model_epoch_42.loom");
// Continues from epoch 43
```

### Example 5: MNIST Complete Example

```cpp
#include "loom/nn/linear.h"
#include "loom/nn/module.h"
#include "loom/training/trainer.h"
#include "loom/dataloader/dataloader.h"
#include "loom/dataloader/mnist_dataset.h"

using namespace loom;

// Model
class MLP : public nn::Module {
    nn::Linear fc1{784, 128};
    nn::Linear fc2{128, 10};
public:
    Tensor forward(const Tensor& x) override {
        x = relu(fc1(x));
        return fc2(x);
    }
};

// Helper: Compute accuracy
float computeAccuracy(const Tensor& logits, const Tensor& targets) {
    auto pred = logits.argmax(1);
    auto true_labels = targets.argmax(1);
    return (pred == true_labels).sum().item() / pred.size(0);
}

int main() {
    // Data
    MNISTDataset train_data("data/train-images.idx3-ubyte", 
                            "data/train-labels.idx1-ubyte");
    MNISTDataset val_data("data/test-images.idx3-ubyte", 
                          "data/test-labels.idx1-ubyte");
    DataLoader train_loader(train_data, 64, true);
    DataLoader val_loader(val_data, 64, false);
    
    // Model
    auto model = std::make_shared<MLP>();
    
    // Loss and optimizer (user-implemented)
    auto loss_fn = createCrossEntropyLoss();
    auto optimizer = createSGD(model->parameters(), 0.01, 0.9);
    
    // Training
    training::Trainer trainer;
    trainer.withTrainLoader(train_loader)
           .withValLoader(val_loader)
           .withForward([&](const Tensor& x) {
               return model->forward(x);
           })
           .withLoss([&](const Tensor& pred, const Tensor& target) {
               Tensor loss = loss_fn->compute(pred, target);
               // Return both loss tensor and scalar for logging
               return loss;
           })
           .withBackward([&](const Tensor& loss) {
               loss.backward();
           })
           .withOptimizerStep([&]() {
               optimizer->step();
               model->zeroGrad();
           })
           .withMetrics(
               [](const Tensor& pred, const Tensor& target) {
                   return std::unordered_map<std::string, float>{
                       {"accuracy", computeAccuracy(pred, target)}
                   };
               }
           )
           .withCheckpoint(
               [&](const std::string& path, size_t epoch) {
                   // Your checkpoint save logic
                   saveModel(path, model, optimizer, epoch);
               },
               [&](const std::string& path) {
                   // Your checkpoint load logic
                   return loadModel(path, model, optimizer);
               },
               "checkpoints/mnist"
           )
           .withTUI();
    
    trainer.train(100);
    
    std::cout << "Training complete!\n";
    std::cout << "Best validation accuracy: " 
              << trainer.history().getBest("accuracy").val_metrics.at("accuracy") 
              << "\n";
    
    return 0;
}
```

---

## Testing Strategy

### Unit Tests

#### `tests/test_metrics.cpp`
- `MetricsSnapshot` creation and access
- `TrainingHistory` recording
- History queries (getHistory, getBest, getAverage)
- JSON/CSV export

#### `tests/test_profiler_bridge.cpp`
- Mock profiler implementation
- `ProfileSnapshot` creation
- Interface compliance

#### `tests/test_trainer.cpp`
- Configuration validation
- Training loop execution
- State management (pause/stop/save)
- Monitor notifications
- Callback invocation

### Integration Tests

#### `tests/integration/test_mnist_training.cpp`
- Train MNIST for 5 epochs with console monitor
- Verify loss decreases
- Verify history is recorded
- Verify checkpoints can be saved/loaded

#### `tests/integration/test_tui_mock.cpp`
- Train with TUI using mock profiler
- Verify TUI displays correct data
- Test user input simulation
- Test view switching

### Manual Testing

- [ ] Train MNIST with TUI for 100 epochs
- [ ] Test pause/resume during training
- [ ] Test save checkpoint during training
- [ ] Test quit with confirmation
- [ ] Test all view switches (TAB)
- [ ] Test on different terminal sizes
- [ ] Test with profiler attached
- [ ] Test without profiler
- [ ] Test resume from checkpoint
- [ ] Test multiple monitors simultaneously

---

## Performance Considerations

### TUI Overhead

- **UI thread**: Separate from training, minimal impact
- **Update frequency**: Throttled to ~10 FPS
- **Lock-free reads**: Minimizes contention
- **Lazy rendering**: Only updates on state change

### Memory

- **Loss history**: Configurable max size (default 100 epochs)
- **Log messages**: Ring buffer (default 20 messages)
- **Profile snapshots**: Not stored, only latest

### Metrics Collection

- **User responsibility**: User controls what's computed
- **Lazy evaluation**: Metrics only computed if monitors attached
- **Batch-level granularity**: Metrics computed per batch (user choice)

---

## Future Enhancements

### Phase 2 Features (After Initial Release)

1. **Additional Monitors**
   - Tensorboard integration
   - Weights & Biases integration
   - CSV logging
   - Remote monitoring (HTTP endpoint)

2. **Advanced TUI Features**
   - Zooming in charts
   - Metric selection/filtering
   - Custom color themes
   - Export screenshot

3. **Training Features**
   - Learning rate scheduling hooks
   - Early stopping
   - Gradient clipping hooks
   - Mixed precision training integration

4. **Profiler Enhancements**
   - GPU profiling (CUDA events)
   - Layer-by-layer timing
   - Memory leak detection
   - Comparative profiling

5. **Distributed Training**
   - Multi-GPU support
   - Distributed training hooks
   - Aggregate metrics across workers

---

## Appendix A: Dependencies

### External Dependencies

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| FTXUI | 5.0.0+ | Terminal UI | MIT |

### Internal Dependencies

| Component | Location | Status |
|-----------|----------|--------|
| Tensor | `loom/tensor/` | ✅ Exists |
| Module | `loom/nn/module.h` | ✅ Exists |
| DataLoader | `loom/dataloader/` | ✅ Exists |
| Logger | `loom/logger.h` | ✅ Exists |
| Device | `loom/device.h` | ✅ Exists |
| Parameter | `loom/nn/parameter.h` | ✅ Exists |

---

## Appendix B: CMake Integration

### Root CMakeLists.txt

```cmake
# Add FTXUI
include(FetchContent)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG v5.0.0
)

set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(ftxui)

# Add training module
add_subdirectory(src/common/training)
```

### src/common/training/CMakeLists.txt

```cmake
set(TRAINING_SOURCES
    metrics.cpp
    trainer.cpp
    tui_monitor.cpp
    console_monitor.cpp
)

add_library(loom_training ${TRAINING_SOURCES})

target_include_directories(loom_training
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(loom_training
    PUBLIC
        loom_tensor
        loom_dataloader
        loom_logger
        loom_nn
        ftxui::screen
        ftxui::dom
        ftxui::component
)
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Trainer** | Orchestrator class that manages the training loop |
| **Monitor** | Interface for observing and interacting with training |
| **TUI** | Terminal User Interface - interactive text-based UI |
| **Profiler** | Tool for measuring timing and memory usage |
| **Snapshot** | Point-in-time capture of metrics or profiling data |
| **Checkpoint** | Saved model state that can be resumed later |
| **Epoch** | One complete pass through the training dataset |
| **Batch** | Subset of data processed in one iteration |
| **Metric** | Measurement of model performance (loss, accuracy, etc.) |
| **Callback** | User-provided function called by the Trainer |

---

## Appendix D: References

- [FTXUI Documentation](https://arthursonzogni.github.io/FTXUI/)
- [PyTorch Trainer Pattern](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Fast.ai Learner](https://docs.fast.ai/learner.html)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-07 | Initial design document |

---

**End of Document**

