"""The picture area, presented in HDR.

A window like ours is SDR: its white is the brightest thing it can show. An HDR capture
shown in it loses everything brighter than that white (the sun, the sky around it, every
light): they are cut to white. Windows' own viewer does not have that limit, because it
presents the capture in HDR. On a screen with HDR switched on, this surface does the same.

It is a native child window with its own Direct3D swap chain in scRGB (linear BT.709,
half floats, 1.0 = 80 nits, as far above 1.0 as the picture goes): the format in which
Windows composes an HDR desktop, and the very format of a JXR capture. It draws

* left of the divider, the capture's own values, untouched: what the game showed;
* right of it, the SDR result at the luminance an SDR window gives it on this screen;
* over them the divider, its handle and the labels, at the luminance of ordinary SDR
  interface (``scale`` = the screen's SDR white / 80).

The shaders are given to Qt as HLSL source, which Direct3D compiles when the surface is
created: no shader-baking tool is needed to build the application.

Everything here needs a real screen. Off screen (tests, self-checks) and on a screen
without HDR, ``hdr_output_available`` is false and the window paints its SDR view instead.
"""
import logging
import struct
import sys
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QGuiApplication, QImage, QRhi, QRhiBuffer, QRhiD3D11InitParams,
                           QRhiDepthStencilClearValue, QRhiGraphicsPipeline, QRhiReadbackDescription,
                           QRhiReadbackResult, QRhiSampler, QRhiShaderResourceBinding, QRhiShaderStage,
                           QRhiSwapChain, QRhiTexture, QRhiTextureSubresourceUploadDescription,
                           QRhiTextureUploadDescription, QRhiTextureUploadEntry, QRhiVertexInputAttribute,
                           QRhiVertexInputBinding, QRhiVertexInputLayout, QRhiViewport, QShader, QShaderCode,
                           QShaderKey, QShaderVersion, QSurface, QWindow)

from ..display import sdr_white_nits

SCRGB_WHITE_NITS = 80.0

_VERTEX = b"""
struct Input { float2 position : TEXCOORD0; float2 uv : TEXCOORD1; };
struct Output { float4 position : SV_Position; float2 uv : TEXCOORD0; };
Output main(Input input)
{
    Output output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}
"""

# Colours on this surface are LINEAR and absolute: 1.0 is 80 nits. The two pictures arrive
# that way. The interface drawn over them (divider, handle, labels) arrives as ordinary
# sRGB pixels: it is decoded, then multiplied by `view.y` (the screen's SDR white / 80) so
# that it is as bright as the rest of the window.
_FRAGMENT = b"""
cbuffer buf : register(b0)
{
    float4 view;      // split (0..1), SDR scale, half width of the divider (px), size of the handle (px)
    float4 place;     // the picture's rectangle on the surface, in pixels: x, y, width, height
    float4 divider;   // colour of the divider (sRGB) and its opacity
};
Texture2D beforeTexture : register(t1);   SamplerState beforeSampler : register(s1);
Texture2D afterTexture : register(t2);    SamplerState afterSampler : register(s2);
Texture2D overlayTexture : register(t3);  SamplerState overlaySampler : register(s3);
Texture2D handleTexture : register(t4);   SamplerState handleSampler : register(s4);

float3 light(float3 encoded)
{
    float3 low = encoded / 12.92;
    float3 high = pow(max((encoded + 0.055) / 1.055, 0.0), 2.4);
    return lerp(low, high, step(0.04045, encoded));
}

float3 over(float3 under, float4 premultiplied, float scale)
{
    float alpha = premultiplied.a;
    float3 colour = light(premultiplied.rgb / max(alpha, 1e-4)) * scale;
    return under * (1.0 - alpha) + colour * alpha;
}

float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 pixel = position.xy - place.xy;
    float x = view.x * place.z;
    float3 before = beforeTexture.SampleLevel(beforeSampler, uv, 0.0).rgb;
    float3 after = afterTexture.SampleLevel(afterSampler, uv, 0.0).rgb;
    float3 colour = pixel.x < x ? before : after;
    float covered = saturate(view.z + 0.5 - abs(pixel.x - x)) * divider.a;
    colour = lerp(colour, light(divider.rgb) * view.y, covered);
    float2 h = (pixel - float2(x, place.w * 0.5)) / view.w + 0.5;
    float4 handle = handleTexture.SampleLevel(handleSampler, saturate(h), 0.0);
    float inside = step(0.0, h.x) * step(h.x, 1.0) * step(0.0, h.y) * step(h.y, 1.0);
    colour = over(colour, handle * inside, view.y);
    colour = over(colour, overlayTexture.SampleLevel(overlaySampler, uv, 0.0), view.y);
    return float4(colour, 1.0);
}
"""


def hdr_output_available() -> bool:
    """Can the picture area be presented in HDR? On Windows, on a real screen, with HDR
    switched on for at least one display. Whether the screen the window is on is one of
    them is asked for every picture (see the worker): on a screen without HDR the SDR view
    is used. (The surface also checks the swap chain format itself when it starts, and
    reports a failure rather than showing a wrong picture.)"""
    if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
        return False
    return sdr_white_nits()[1] == "display"


def _shader(stage, source: bytes) -> QShader:
    shader = QShader()
    shader.setStage(stage)
    shader.setShader(QShaderKey(QShader.Source.HlslShader, QShaderVersion(50)), QShaderCode(source, b"main"))
    return shader


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


class HdrSurface(QWindow):
    """See the module. The owner gives it pictures and a layout; it presents them."""

    pointer = Signal(str, float, float)        # "press", "move", "release" or "double"; position on the surface
    key = Signal(int)
    failed = Signal(str)                       # HDR cannot be presented after all: use the SDR view

    def __init__(self):
        super().__init__()
        self.setSurfaceType(QSurface.SurfaceType.Direct3DSurface)
        self._rhi = None
        self._broken = False
        self._before: Optional[np.ndarray] = None
        self._after: Optional[np.ndarray] = None
        self._overlay: Optional[QImage] = None
        self._handle: Optional[QImage] = None
        self._dirty = set()
        self._split = 0.5
        self._scale = 203.0 / SCRGB_WHITE_NITS
        self._place = (0.0, 0.0, 1.0, 1.0)
        self._divider = (0.75, (1.0, 1.0, 1.0, 0.9))
        self._handle_size = 32.0
        self._background = (0.0, 0.0, 0.0)
        self._textures = {}
        self._presented = None
        # A file dragged over the picture lands on this surface, not on the window under it:
        # whoever is named here is told, and answers for it.
        self.drop_target = None

    # ----------------------------------------------------------------- content
    def set_pictures(self, before: np.ndarray, after: np.ndarray) -> None:
        """(h, w, 4) half floats, scRGB: the source and the result, at the size shown."""
        self._before, self._after = before, after
        self._dirty |= {"before", "after"}
        self.requestUpdate()

    def set_overlay(self, image: QImage) -> None:
        """What is drawn over the pictures, the size of their rectangle: ordinary sRGB pixels."""
        self._overlay = image.convertToFormat(QImage.Format_RGBA8888_Premultiplied)
        self._dirty.add("overlay")
        self.requestUpdate()

    def set_handle(self, image: QImage, size_px: float) -> None:
        self._handle = image.convertToFormat(QImage.Format_RGBA8888_Premultiplied)
        self._handle_size = float(size_px)
        self._dirty.add("handle")
        self.requestUpdate()

    def set_split(self, split: float) -> None:
        if split != self._split:
            self._split = float(split)
            self.requestUpdate()

    def set_layout(self, place_px: Tuple[float, float, float, float], white_nits: float,
                   divider_half_width_px: float, divider_rgba, background_srgb) -> None:
        """Where the picture sits on the surface (pixels), what SDR white is on this screen,
        and the colours of what surrounds and divides the pictures (sRGB, 0..1)."""
        self._place = tuple(float(v) for v in place_px)
        self._scale = float(white_nits) / SCRGB_WHITE_NITS
        self._divider = (float(divider_half_width_px), tuple(float(v) for v in divider_rgba))
        self._background = tuple(float(v) for v in background_srgb)
        self.requestUpdate()

    def presents_hdr(self) -> bool:
        return self._rhi is not None and not self._broken

    # ------------------------------------------------------------------ events
    def exposeEvent(self, event) -> None:
        if self.isExposed() and not self._broken:
            if self._rhi is None:
                self._start()
            self._render()

    def event(self, event) -> bool:
        kind = event.type()
        if kind == QEvent.UpdateRequest:
            if self.isExposed() and not self._broken:
                self._render()
            return True
        if kind in (QEvent.DragEnter, QEvent.DragMove, QEvent.DragLeave, QEvent.Drop) and self.drop_target is not None:
            QCoreApplication.sendEvent(self.drop_target, event)
            return True
        if kind == QEvent.PlatformSurface and event.surfaceEventType() == event.SurfaceEventType.SurfaceAboutToBeDestroyed:
            self._release()
        return super().event(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.pointer.emit("press", event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:
        self.pointer.emit("move", event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event) -> None:
        self.pointer.emit("release", event.position().x(), event.position().y())

    def mouseDoubleClickEvent(self, event) -> None:
        self.pointer.emit("double", event.position().x(), event.position().y())

    def keyPressEvent(self, event) -> None:
        self.key.emit(event.key())

    # --------------------------------------------------------------- rendering
    def _fail(self, why: str) -> None:
        logging.warning("HDR surface unavailable: %s", why)
        self._broken = True
        self._release()
        self.failed.emit(why)

    def _start(self) -> None:
        try:
            self._rhi = QRhi.create(QRhi.Implementation.D3D11, QRhiD3D11InitParams())
            if self._rhi is None:
                return self._fail("Direct3D 11 could not be started")
            rhi = self._rhi
            self._swap = rhi.newSwapChain()
            self._swap.setWindow(self)
            if not self._swap.isFormatSupported(QRhiSwapChain.Format.HDRExtendedSrgbLinear) \
                    or not rhi.isTextureFormatSupported(QRhiTexture.Format.RGBA16F):
                return self._fail("this display does not take an scRGB swap chain")
            self._swap.setFormat(QRhiSwapChain.Format.HDRExtendedSrgbLinear)
            self._swap.setFlags(QRhiSwapChain.Flag.UsedAsTransferSource)       # so that a frame can be read back
            self._pass = self._swap.newCompatibleRenderPassDescriptor()
            self._swap.setRenderPassDescriptor(self._pass)
            if not self._swap.createOrResize():
                return self._fail("the swap chain could not be created")
            # one quad over the picture's rectangle: position (clip space), then texture coordinates
            self._quad = struct.pack("16f", -1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0)
            self._vertices = rhi.newBuffer(QRhiBuffer.Type.Immutable, QRhiBuffer.UsageFlag.VertexBuffer, len(self._quad))
            self._uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, 48)
            self._sampler = rhi.newSampler(QRhiSampler.Filter.Linear, QRhiSampler.Filter.Linear, QRhiSampler.Filter.None_,
                                           QRhiSampler.AddressMode.ClampToEdge, QRhiSampler.AddressMode.ClampToEdge)
            if not (self._vertices.create() and self._uniforms.create() and self._sampler.create()):
                return self._fail("buffers could not be created")
            for name, pixel_format in (("before", QRhiTexture.Format.RGBA16F), ("after", QRhiTexture.Format.RGBA16F),
                                       ("overlay", QRhiTexture.Format.RGBA8), ("handle", QRhiTexture.Format.RGBA8)):
                texture = rhi.newTexture(pixel_format, QSize(1, 1))
                if not texture.create():
                    return self._fail("textures could not be created")
                self._textures[name] = texture
            self._bindings = rhi.newShaderResourceBindings()
            self._bind()
            self._pipeline = rhi.newGraphicsPipeline()
            self._pipeline.setShaderStages([
                QRhiShaderStage(QRhiShaderStage.Type.Vertex, _shader(QShader.Stage.VertexStage, _VERTEX)),
                QRhiShaderStage(QRhiShaderStage.Type.Fragment, _shader(QShader.Stage.FragmentStage, _FRAGMENT))])
            layout = QRhiVertexInputLayout()
            layout.setBindings([QRhiVertexInputBinding(16)])
            layout.setAttributes([QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute.Format.Float2, 0),
                                  QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute.Format.Float2, 8)])
            self._pipeline.setVertexInputLayout(layout)
            self._pipeline.setTopology(QRhiGraphicsPipeline.Topology.TriangleStrip)
            self._pipeline.setShaderResourceBindings(self._bindings)
            self._pipeline.setRenderPassDescriptor(self._pass)
            if not self._pipeline.create():
                return self._fail("the shaders could not be compiled")
            logging.info("HDR surface started: the picture area is presented in scRGB (1.0 = 80 nits)")
            self._dirty |= {"quad", "before", "after", "overlay", "handle"}
        except Exception as exc:                               # never leave the window without a picture
            self._fail(str(exc))

    def _bind(self) -> None:
        stage = QRhiShaderResourceBinding.StageFlag.FragmentStage
        self._bindings.setBindings(
            [QRhiShaderResourceBinding.uniformBuffer(0, stage, self._uniforms)]
            + [QRhiShaderResourceBinding.sampledTexture(i + 1, stage, self._textures[name], self._sampler)
               for i, name in enumerate(("before", "after", "overlay", "handle"))])
        self._bindings.create()

    def _fit(self, name: str, width: int, height: int) -> bool:
        """Give a texture the size of its picture. True if it had to be made again."""
        texture = self._textures[name]
        if texture.pixelSize() == QSize(width, height):
            return False
        texture.setPixelSize(QSize(width, height))
        texture.create()
        return True

    def _uploads(self, batch) -> None:
        remade = False
        entries = []
        for name, picture in (("before", self._before), ("after", self._after)):
            if name in self._dirty and picture is not None:
                remade |= self._fit(name, picture.shape[1], picture.shape[0])
                entries.append((name, QRhiTextureSubresourceUploadDescription(picture.tobytes())))
        for name, image in (("overlay", self._overlay), ("handle", self._handle)):
            if name in self._dirty and image is not None:
                remade |= self._fit(name, image.width(), image.height())
                entries.append((name, QRhiTextureSubresourceUploadDescription(image)))
        if remade:
            self._bind()
        for name, description in entries:
            batch.uploadTexture(self._textures[name], QRhiTextureUploadDescription(QRhiTextureUploadEntry(0, 0, description)))
        if "quad" in self._dirty:
            batch.uploadStaticBuffer(self._vertices, self._quad)
        self._dirty.clear()

    def _render(self, read_back: bool = False) -> None:
        if self._rhi is None or self._broken:
            return
        try:
            if self._swap.currentPixelSize() != self._swap.surfacePixelSize():
                if not self._swap.createOrResize():
                    return
            if self._rhi.beginFrame(self._swap) != QRhi.FrameOpResult.FrameOpSuccess:
                return
            size = self._swap.currentPixelSize()
            x, y, w, h = self._place
            half_width, (r, g, b, a) = self._divider
            batch = self._rhi.nextResourceUpdateBatch()
            self._uploads(batch)
            batch.updateDynamicBuffer(self._uniforms, 0, 48, struct.pack(
                "12f", self._split, self._scale, half_width, self._handle_size, x, y, w, h, r, g, b, a))
            clear = QColor.fromRgbF(*[min(1.0, _srgb_to_linear(c) * self._scale) for c in self._background])
            commands = self._swap.currentFrameCommandBuffer()
            commands.beginPass(self._swap.currentFrameRenderTarget(), clear, QRhiDepthStencilClearValue(1.0, 0), batch)
            if self._before is not None and w >= 1 and h >= 1:
                commands.setGraphicsPipeline(self._pipeline)
                # viewports are counted from the bottom of the surface
                commands.setViewport(QRhiViewport(x, size.height() - (y + h), w, h))
                commands.setShaderResources()
                commands.setVertexInput(0, [(self._vertices, 0)])
                commands.draw(4)
            result = None
            if read_back:
                after = self._rhi.nextResourceUpdateBatch()
                result = QRhiReadbackResult()
                after.readBackTexture(QRhiReadbackDescription(), result)
                commands.endPass(after)
            else:
                commands.endPass()
            self._rhi.endFrame(self._swap)
            if result is not None:
                self._rhi.finish()
                size = result.pixelSize
                data = bytes(result.data)
                self._presented = np.frombuffer(data, dtype=np.float16).reshape(size.height(), size.width(), 4).copy()
        except Exception as exc:
            self._fail(str(exc))

    def read_back(self) -> Optional[np.ndarray]:
        """What the surface presents, read from its display buffer: (H, W, 4) half floats,
        scRGB. For checking, not for the interface."""
        self._presented = None
        self._render(read_back=True)
        return self._presented

    def _release(self) -> None:
        """Let go of everything Direct3D holds, in order, before the window goes."""
        for name in ("_pipeline", "_bindings", "_sampler", "_uniforms", "_vertices"):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.destroy()
                setattr(self, name, None)
        for texture in self._textures.values():
            texture.destroy()
        self._textures = {}
        for name in ("_pass", "_swap"):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.destroy()
                setattr(self, name, None)
        self._rhi = None
