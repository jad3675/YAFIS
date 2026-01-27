"""
GPU Resource Wrappers - Textures and Buffers for persistent GPU memory.

These classes wrap GPU resources to enable efficient multi-stage processing
without intermediate CPU readbacks.
"""

from typing import Optional, Tuple, Any
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)


class GPUTexture:
    """
    GPU texture wrapper for image data.
    
    Uses rgba32float format for high-dynamic-range processing.
    Data stays on GPU between operations - only readback at the end.
    """
    
    def __init__(self, width: int, height: int, usage: int = 0) -> None:
        """
        Create a GPU texture.
        
        Args:
            width: Texture width in pixels
            height: Texture height in pixels  
            usage: wgpu texture usage flags (auto-configured if 0)
        """
        from .gpu_device import GPUDevice
        import wgpu
        
        self.width = width
        self.height = height
        self.format = "rgba32float"
        
        gpu = GPUDevice.get()
        if not gpu.is_wgpu or not gpu.wgpu_device:
            raise RuntimeError("wgpu device required for GPUTexture")
        
        # Default usage: can be sampled, stored to, and copied from/to
        if usage == 0:
            usage = (
                wgpu.TextureUsage.TEXTURE_BINDING |
                wgpu.TextureUsage.STORAGE_BINDING |
                wgpu.TextureUsage.COPY_DST |
                wgpu.TextureUsage.COPY_SRC
            )
        
        self._texture = gpu.wgpu_device.create_texture(
            size=(width, height, 1),
            format=self.format,
            usage=usage
        )
        self._view = self._texture.create_view()
    
    @property
    def texture(self) -> Any:
        """Get the underlying wgpu texture."""
        return self._texture
    
    @property
    def view(self) -> Any:
        """Get the texture view for binding."""
        return self._view
    
    def upload(self, data: np.ndarray) -> None:
        """
        Upload numpy array to GPU texture.
        
        Args:
            data: Image data as float32 array, shape (H, W, 3) or (H, W, 4)
        """
        from .gpu_device import GPUDevice
        
        gpu = GPUDevice.get()
        if not gpu.wgpu_device:
            return
        
        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Ensure RGBA (add alpha channel if RGB)
        if data.ndim == 2:
            # Grayscale to RGBA
            rgba = np.ones((data.shape[0], data.shape[1], 4), dtype=np.float32)
            rgba[:, :, 0] = data
            rgba[:, :, 1] = data
            rgba[:, :, 2] = data
            data = rgba
        elif data.shape[2] == 3:
            # RGB to RGBA
            rgba = np.ones((data.shape[0], data.shape[1], 4), dtype=np.float32)
            rgba[:, :, :3] = data
            data = rgba
        
        # Ensure contiguous
        data = np.ascontiguousarray(data)
        
        # Upload to GPU
        # bytes_per_row = width * 4 channels * 4 bytes per float = width * 16
        gpu.wgpu_device.queue.write_texture(
            {"texture": self._texture},
            data,
            {"bytes_per_row": data.shape[1] * 16, "rows_per_image": data.shape[0]},
            (data.shape[1], data.shape[0], 1)
        )
    
    def readback(self) -> np.ndarray:
        """
        Download texture data from GPU to CPU.
        
        Returns:
            float32 numpy array of shape (H, W, 4)
        """
        from .gpu_device import GPUDevice
        import wgpu
        
        gpu = GPUDevice.get()
        if not gpu.wgpu_device or not self._texture:
            return np.zeros((self.height, self.width, 4), dtype=np.float32)
        
        # Calculate aligned bytes per row (must be multiple of 256)
        bytes_per_row = (self.width * 16 + 255) & ~255
        buffer_size = bytes_per_row * self.height
        
        # Create staging buffer for readback
        staging = gpu.wgpu_device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        
        # Copy texture to staging buffer
        encoder = gpu.wgpu_device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": self._texture},
            {"buffer": staging, "bytes_per_row": bytes_per_row},
            (self.width, self.height, 1)
        )
        gpu.wgpu_device.queue.submit([encoder.finish()])
        
        # Map and read
        staging.map_sync(mode=wgpu.MapMode.READ)
        raw_data = staging.read_mapped()
        
        # Parse the data accounting for row alignment
        arr = np.frombuffer(raw_data, dtype=np.float32).reshape(
            (self.height, bytes_per_row // 4)
        )
        # Extract actual pixel data (remove padding)
        pixels = arr[:, :self.width * 4].reshape((self.height, self.width, 4))
        
        result = pixels.copy()
        staging.destroy()
        
        return result
    
    def destroy(self) -> None:
        """Release GPU resources."""
        try:
            self._view = None
            if self._texture:
                self._texture.destroy()
                self._texture = None
        except Exception:
            pass


class GPUBuffer:
    """
    GPU buffer wrapper for uniform and storage data.
    """
    
    def __init__(self, size: int, usage: int) -> None:
        """
        Create a GPU buffer.
        
        Args:
            size: Buffer size in bytes
            usage: wgpu buffer usage flags
        """
        from .gpu_device import GPUDevice
        
        gpu = GPUDevice.get()
        if not gpu.is_wgpu or not gpu.wgpu_device:
            raise RuntimeError("wgpu device required for GPUBuffer")
        
        self.size = size
        self._buffer = gpu.wgpu_device.create_buffer(size=size, usage=usage)
    
    @property
    def buffer(self) -> Any:
        """Get the underlying wgpu buffer."""
        return self._buffer
    
    def upload(self, data: np.ndarray) -> None:
        """Upload data to buffer."""
        from .gpu_device import GPUDevice
        
        gpu = GPUDevice.get()
        if not gpu.wgpu_device:
            return
        
        gpu.wgpu_device.queue.write_buffer(self._buffer, 0, data.tobytes())
    
    def upload_bytes(self, data: bytes) -> None:
        """Upload raw bytes to buffer."""
        from .gpu_device import GPUDevice
        
        gpu = GPUDevice.get()
        if not gpu.wgpu_device:
            return
        
        gpu.wgpu_device.queue.write_buffer(self._buffer, 0, data)
    
    def destroy(self) -> None:
        """Release GPU resources."""
        try:
            if self._buffer:
                self._buffer.destroy()
                self._buffer = None
        except Exception:
            pass


class TexturePool:
    """
    Pool of reusable GPU textures to avoid repeated allocation.
    
    Note: Each unique (width, height, usage, label) combination gets its own texture.
    The label is important to prevent the same texture being used for input and output.
    """
    
    def __init__(self) -> None:
        self._pool: dict[Tuple[int, int, int, str], GPUTexture] = {}
    
    def get(self, width: int, height: int, usage: int, label: str = "") -> GPUTexture:
        """
        Get or create a texture from the pool.
        
        Args:
            width: Texture width
            height: Texture height
            usage: wgpu usage flags
            label: Label to distinguish textures with same dimensions
            
        Returns:
            GPUTexture instance
        """
        key = (width, height, usage, label)
        if key not in self._pool:
            self._pool[key] = GPUTexture(width, height, usage)
            logger.debug(f"Created pooled texture: {width}x{height} ({label})")
        return self._pool[key]
    
    def clear(self) -> None:
        """Release all pooled textures."""
        for tex in self._pool.values():
            tex.destroy()
        self._pool.clear()
    
    def __len__(self) -> int:
        return len(self._pool)
