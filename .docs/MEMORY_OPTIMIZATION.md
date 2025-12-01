# Handling Large Point Clouds - Memory Optimization Guide

## Problem

When processing large point clouds (e.g., KITTI dataset with ~120k points per frame), the preprocessing step would fail with a memory allocation error:

```
Error: Unable to allocate 116. GiB for an array with shape (124627, 124627) and data type float64
```

This occurred because the statistical outlier removal method was creating a full pairwise distance matrix using `scipy.spatial.distance.cdist`, which requires O(N²) memory.

## Solution

The following improvements have been implemented:

### 1. Memory-Efficient Nearest Neighbor Search

Replaced the `cdist` approach with KD-tree based search:

- **Before**: Created full N×N distance matrix (116 GiB for 124k points)
- **After**: Used `scipy.spatial.KDTree` for efficient nearest neighbor queries (minimal memory)

### 2. Automatic Safety Limit

Added `max_points_for_outliers` parameter (default: 50,000):

- Skips outlier removal for point clouds exceeding this limit
- Prints a warning suggesting to use voxel downsampling first

### 3. Voxel Downsampling

The existing voxel downsampling feature is now the recommended first step for large datasets:

- Reduces point cloud size before outlier removal
- Example: `--voxel-size 0.1` reduces 120k points to ~20-30k points

## Usage Recommendations

### For KITTI or Similar Large Datasets

```bash
# Option 1: Use the pre-configured KITTI config
roboqa path/to/bag_file.db3 --config examples/config_kitti.yaml

# Option 2: Use CLI arguments
roboqa path/to/bag_file.db3 --voxel-size 0.1 --max-points-for-outliers 50000

# Option 3: Disable outlier removal if downsampling is sufficient
roboqa path/to/bag_file.db3 --voxel-size 0.1 --no-outlier-removal
```

### Configuration File Example

```yaml
preprocessing:
  # Downsample to ~20-30k points
  voxel_size: 0.1
  
  # Remove outliers after downsampling
  remove_outliers: true
  
  # Safety limit (outlier removal skipped if exceeded)
  max_points_for_outliers: 50000
  
  outlier_method: statistical
  outlier_params:
    nb_neighbors: 20
    std_ratio: 2.5
```

## Technical Details

### Memory Complexity Comparison

| Method | Memory | Time | Suitable For |
|--------|--------|------|-------------|
| `cdist` (old) | O(N²) | O(N²) | Small clouds (<5k points) |
| `KDTree` (new) | O(N) | O(N log N) | Any size |

### KD-Tree Implementation

The statistical outlier removal now uses:

```python
tree = KDTree(points)
distances, _ = tree.query(points, k=nb_neighbors + 1)
```

This provides:
- Constant memory overhead per query
- Logarithmic time complexity for nearest neighbor search
- Suitable for point clouds of any size

### Radius-Based Outlier Removal

Also updated to use KD-tree:

```python
tree = KDTree(points)
neighbor_counts = [len(tree.query_ball_point(point, radius)) - 1 for point in points]
```

## Performance Impact

For a typical KITTI point cloud:

- **Before**: Crash with 116 GiB allocation error
- **After** (with voxel_size=0.1):
  - Downsampling: ~1-2 seconds per frame
  - Outlier removal: ~0.5 seconds per frame
  - Total memory: <1 GiB

## Future Improvements

Potential optimizations for consideration:

1. Parallel processing of frames
2. Adaptive voxel size based on point cloud density
3. Streaming processing for very large datasets
