import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
import math
import numpy as np
import blf
from typing import Optional, Tuple
from bpy.props import FloatProperty, IntProperty, PointerProperty

# ==============================================================================
# 1. CONSTANTS & DOMAIN CONFIGURATION
# ==============================================================================

class LenticularDefaults:
    COLS: int = 8
    ROWS: int = 6
    TILT: float = -0.16667
    PITCH: float = 49.83
    CENTER: float = 0.5
    SUBP: float = 1.0 / (3.0 * 3840.0)
    VIEW_PORTION: float = 1.0
    FOCUS: float = 0.0
    BASELINE: float = 0.25
    FOCAL_DISTANCE: float = 2.5
    TILE_SIZE: int = 256
    BUFFER_NAME: str = "Quilt_Buffer"
    RIG_NAME: str = "Lenticular_Rig"
    COLLECTION_NAME: str = "Lenticular_Cameras"


def _on_rig_parameter_change(self, context: bpy.types.Context) -> None:
    """Callback to rebuild camera array when baseline or convergence shifts."""
    if bpy.data.objects.get(LenticularDefaults.RIG_NAME):
        bpy.ops.object.create_lenticular_rig('EXEC_DEFAULT')


class LenticularSettings(bpy.types.PropertyGroup):
    cols: IntProperty(
        name="Columns",
        default=LenticularDefaults.COLS,
        min=1,
        max=16,
        update=_on_rig_parameter_change,
    )
    rows: IntProperty(
        name="Rows",
        default=LenticularDefaults.ROWS,
        min=1,
        max=16,
        update=_on_rig_parameter_change,
    )
    tilt: FloatProperty(name="Tilt", default=LenticularDefaults.TILT, precision=5)
    pitch: FloatProperty(name="Pitch", default=LenticularDefaults.PITCH, precision=4)
    center: FloatProperty(
        name="Phase Center",
        default=LenticularDefaults.CENTER,
        min=0.0,
        max=1.0,
        precision=4,
    )
    subp: FloatProperty(
        name="Subpixel Delta", default=LenticularDefaults.SUBP, precision=7
    )
    view_portion: FloatProperty(
        name="View Portion",
        default=LenticularDefaults.VIEW_PORTION,
        min=0.1,
        max=1.0,
    )
    focus: FloatProperty(
        name="Dynamic Focus",
        description="Focal plane zero-parallax disparity offset",
        default=LenticularDefaults.FOCUS,
        min=-0.5,
        max=0.5,
        step=0.01,
        precision=4,
    )
    baseline: FloatProperty(
        name="Baseline Track",
        default=LenticularDefaults.BASELINE,
        min=0.001,
        unit='LENGTH',
        update=_on_rig_parameter_change,
    )
    focal_distance: FloatProperty(
        name="Convergence Distance",
        default=LenticularDefaults.FOCAL_DISTANCE,
        min=0.01,
        unit='LENGTH',
        update=_on_rig_parameter_change,
    )


# ==============================================================================
# 2. PROJECTION & CAMERA MATHEMATICS SERVICE
# ==============================================================================

class CameraProjectionBuilder:
    """Calculates non-distorting off-axis OpenGL projection matrices with lens shift."""

    @staticmethod
    def calculate(camera_obj: bpy.types.Object, width: int, height: int) -> Matrix:
        cam = camera_obj.data
        near = cam.clip_start
        far = cam.clip_end
        aspect = width / height
        sensor_x = cam.sensor_width
        sensor_y = cam.sensor_height

        fit = cam.sensor_fit
        if fit == 'AUTO':
            fit = 'HORIZONTAL' if width >= height else 'VERTICAL'

        if fit == 'HORIZONTAL':
            fov_x = 2.0 * math.atan((sensor_x / 2.0) / cam.lens)
            fov_y = 2.0 * math.atan((sensor_x / aspect / 2.0) / cam.lens)
            half_w = near * math.tan(fov_x / 2.0)
            half_h = near * math.tan(fov_y / 2.0)
            shift_x = 2.0 * half_w * cam.shift_x
            shift_y = 2.0 * half_w * cam.shift_y
        else:
            fov_y = 2.0 * math.atan((sensor_y / 2.0) / cam.lens)
            fov_x = 2.0 * math.atan((sensor_y * aspect / 2.0) / cam.lens)
            half_w = near * math.tan(fov_x / 2.0)
            half_h = near * math.tan(fov_y / 2.0)
            shift_x = 2.0 * half_h * cam.shift_x
            shift_y = 2.0 * half_h * cam.shift_y

        left = -half_w + shift_x
        right = half_w + shift_x
        bottom = -half_h + shift_y
        top = half_h + shift_y

        return Matrix((
            ((2.0 * near) / (right - left), 0.0, (right + left) / (right - left), 0.0),
            (0.0, (2.0 * near) / (top - bottom), (top + bottom) / (top - bottom), 0.0),
            (0.0, 0.0, -(far + near) / (far - near), -(2.0 * far * near) / (far - near)),
            (0.0, 0.0, -1.0, 0.0),
        ))


# ==============================================================================
# 3. CAMERA RIG MANAGEMENT SERVICE
# ==============================================================================

class CameraRigService:
    """Manages the parallel off-axis camera array hierarchy in the active scene."""

    @classmethod
    def get_or_create_root(cls, scene: bpy.types.Scene, focal_distance: float) -> bpy.types.Object:
        root = bpy.data.objects.get(LenticularDefaults.RIG_NAME)
        if not root:
            root = bpy.data.objects.new(LenticularDefaults.RIG_NAME, None)
            root.empty_display_type = 'ARROWS'
            scene.collection.objects.link(root)
        root.location = (0.0, -focal_distance, 0.0)
        return root

    @classmethod
    def get_or_create_collection(cls, scene: bpy.types.Scene) -> bpy.types.Collection:
        col = bpy.data.collections.get(LenticularDefaults.COLLECTION_NAME)
        if not col:
            col = bpy.data.collections.new(LenticularDefaults.COLLECTION_NAME)
            scene.collection.children.link(col)
        return col

    @classmethod
    def purge_existing_cameras(cls, collection: bpy.types.Collection) -> None:
        for obj in list(collection.objects):
            cam_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if cam_data:
                bpy.data.cameras.remove(cam_data)

    @classmethod
    def build_rig(cls, context: bpy.types.Context) -> int:
        cfg = context.scene.lenticular_settings
        total_views = cfg.cols * cfg.rows
        scene = context.scene

        root = cls.get_or_create_root(scene, cfg.focal_distance)
        collection = cls.get_or_create_collection(scene)
        cls.purge_existing_cameras(collection)

        for i in range(total_views):
            cam_data = bpy.data.cameras.new(f"LentiCam_{i:02d}")
            cam_data.display_size = 0.04
            cam_obj = bpy.data.objects.new(f"LentiCam_{i:02d}", cam_data)
            collection.objects.link(cam_obj)
            cam_obj.parent = root

            normalized_step = (i / (total_views - 1)) - 0.5 if total_views > 1 else 0.0
            x_pos = normalized_step * cfg.baseline
            cam_obj.location = (x_pos, 0.0, 0.0)
            cam_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

            fov = cam_data.angle
            cam_data.shift_x = -x_pos / (2.0 * cfg.focal_distance * math.tan(fov / 2.0))

        return total_views


# ==============================================================================
# 4. VIEWPORT CONTEXT & MATERIAL PREPARATION SERVICES
# ==============================================================================

class MaterialColorSynchronizer:
    """Ensures shader node color outputs match viewport diffuse display color."""

    @staticmethod
    def sync() -> None:
        for mat in bpy.data.materials:
            if not mat.use_nodes or not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'EMISSION' and 'Color' in node.inputs:
                    mat.diffuse_color = node.inputs['Color'].default_value
                    break
                elif node.type == 'BSDF_PRINCIPLED' and 'Base Color' in node.inputs:
                    mat.diffuse_color = node.inputs['Base Color'].default_value
                    break


class ViewportStateContext:
    """Context manager to cleanly suppress viewport overlays, 3D cursor, and gizmos."""

    def __init__(self, space: Optional[bpy.types.SpaceView3D]):
        self.space = space
        self.orig_overlays: bool = True
        self.orig_gizmos: bool = True
        self.orig_color_type: str = 'MATERIAL'

    def __enter__(self) -> "ViewportStateContext":
        if self.space:
            self.orig_overlays = self.space.overlay.show_overlays
            self.orig_gizmos = self.space.show_gizmo
            self.orig_color_type = self.space.shading.color_type

            self.space.overlay.show_overlays = False
            self.space.show_gizmo = False
            if self.space.shading.type == 'SOLID':
                self.space.shading.color_type = 'MATERIAL'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.space:
            self.space.overlay.show_overlays = self.orig_overlays
            self.space.show_gizmo = self.orig_gizmos
            if self.space.shading.type == 'SOLID':
                self.space.shading.color_type = self.orig_color_type


# ==============================================================================
# 5. HARDWARE GPU QUILT BAKER SERVICE
# ==============================================================================

class QuiltBakerService:
    """Renders all camera views offscreen and constructs the light-field quilt buffer."""

    @staticmethod
    def get_or_create_buffer(width: int, height: int) -> bpy.types.Image:
        img = bpy.data.images.get(LenticularDefaults.BUFFER_NAME)
        if not img or img.size[0] != width or img.size[1] != height:
            if img:
                bpy.data.images.remove(img)
            img = bpy.data.images.new(
                LenticularDefaults.BUFFER_NAME, width=width, height=height, alpha=True
            )
        return img

    @staticmethod
    def locate_view3d_region(
        context: bpy.types.Context,
    ) -> Tuple[Optional[bpy.types.SpaceView3D], Optional[bpy.types.Region]]:
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    for reg in area.regions:
                        if reg.type == 'WINDOW':
                            return area.spaces.active, reg
        return None, None

    @classmethod
    def bake(cls, context: bpy.types.Context) -> Tuple[int, int, int]:
        cfg = context.scene.lenticular_settings
        total_views = cfg.cols * cfg.rows
        cam_col = bpy.data.collections.get(LenticularDefaults.COLLECTION_NAME)

        if not cam_col or len(cam_col.objects) != total_views:
            CameraRigService.build_rig(context)
            cam_col = bpy.data.collections.get(LenticularDefaults.COLLECTION_NAME)

        space_3d, region = cls.locate_view3d_region(context)
        if not space_3d or not region:
            raise RuntimeError("An active 3D Viewport window is required for capture.")

        MaterialColorSynchronizer.sync()

        tile_size = LenticularDefaults.TILE_SIZE
        quilt_w = cfg.cols * tile_size
        quilt_h = cfg.rows * tile_size
        quilt_img = cls.get_or_create_buffer(quilt_w, quilt_h)
        quilt_arr = np.zeros((quilt_h, quilt_w, 4), dtype=np.float32)

        depsgraph = context.evaluated_depsgraph_get()
        offscreen = gpu.types.GPUOffScreen(tile_size, tile_size)
        fbo = gpu.types.GPUFrameBuffer(color_slots=[offscreen.texture_color])
        cameras = sorted(cam_col.objects, key=lambda c: c.name)

        with ViewportStateContext(space_3d):
            try:
                for idx, cam in enumerate(cameras):
                    cam_eval = cam.evaluated_get(depsgraph)
                    view_mat = cam_eval.matrix_world.inverted()
                    proj_mat = CameraProjectionBuilder.calculate(
                        cam_eval, tile_size, tile_size
                    )

                    offscreen.draw_view3d(
                        context.scene,
                        context.view_layer,
                        space_3d,
                        region,
                        view_mat,
                        proj_mat,
                        do_color_management=True,
                    )

                    with fbo.bind():
                        tile_raw = fbo.read_color(0, 0, tile_size, tile_size, 4, 0, 'FLOAT')
                    tile_np = np.frombuffer(tile_raw, dtype=np.float32).reshape(
                        (tile_size, tile_size, 4)
                    )

                    col_idx = idx % cfg.cols
                    row_idx = cfg.rows - 1 - (idx // cfg.cols)
                    y_start = row_idx * tile_size
                    x_start = col_idx * tile_size

                    quilt_arr[y_start : y_start + tile_size, x_start : x_start + tile_size, :] = tile_np
            finally:
                offscreen.free()

        quilt_img.pixels.foreach_set(quilt_arr.ravel())
        quilt_img.update()
        return total_views, quilt_w, quilt_h


# ==============================================================================
# 6. GPU SHADER PIPELINE SERVICE
# ==============================================================================

class LenticularShaderPipeline:
    """Manages GLSL shaders for real-time lenticular interlacing and debug passes."""

    _instance: Optional["LenticularShaderPipeline"] = None

    def __init__(self):
        self.interlace_shader = self._build_interlace_shader()
        self.blit_shader = self._build_blit_shader()
        self.quad_batch = batch_for_shader(
            self.interlace_shader,
            'TRI_FAN',
            {
                "position": [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
                "uv_in": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            },
        )

    @classmethod
    def get(cls) -> "LenticularShaderPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _build_interlace_shader() -> gpu.types.GPUShader:
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC2', "position")
        info.vertex_in(1, 'VEC2', "uv_in")

        interface = gpu.types.GPUStageInterfaceInfo("interlace_stage")
        interface.smooth('VEC2', "uv")
        info.vertex_out(interface)

        info.sampler(0, 'FLOAT_2D', "quiltTexture")
        info.push_constant('INT', "rows")
        info.push_constant('INT', "cols")
        info.push_constant('FLOAT', "tilt")
        info.push_constant('FLOAT', "pitch")
        info.push_constant('FLOAT', "center")
        info.push_constant('FLOAT', "viewPortionElement")
        info.push_constant('FLOAT', "subp")
        info.push_constant('FLOAT', "focus")
        info.fragment_out(0, 'VEC4', "fragColor")

        info.vertex_source("""
        void main() {
            uv = uv_in;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """)

        info.fragment_source("""
        void main() {
            float newTilt = -tilt;
            int views = cols * rows;
            vec2 viewPortion = vec2(viewPortionElement);
            vec4 rgb[3];

            for (int i = 0; i < 3; i++) {
                float nuv_z = (uv.x + float(i) * subp + uv.y * newTilt) * pitch - center;
                nuv_z = 1.0 - fract(nuv_z);

                float z = floor(nuv_z * float(views));
                z = mod(z - float(cols - 1), float(views));
                if (z < 0.0) z += float(views);

                float focusMod = focus * (1.0 - 2.0 * clamp(z / float(views), 0.0, 1.0));
                
                int zi = int(z);
                int col = zi % cols;
                int row = rows - 1 - (zi / cols);
                float tileIndex = float(col + row * cols);

                float x = (mod(tileIndex, float(cols)) + clamp(uv.x + focusMod, 0.0, 1.0)) / float(cols);
                float y = (floor(tileIndex / float(cols)) + uv.y) / float(rows);
                vec2 sampleCoord = vec2(x * viewPortion.x, y * viewPortion.y);

                rgb[i] = texture(quiltTexture, sampleCoord);
            }

            fragColor = vec4(rgb[0].r, rgb[1].g, rgb[2].b, 1.0);
        }
        """)
        return gpu.shader.create_from_info(info)

    @staticmethod
    def _build_blit_shader() -> gpu.types.GPUShader:
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC2', "position")
        info.vertex_in(1, 'VEC2', "uv_in")

        interface = gpu.types.GPUStageInterfaceInfo("blit_stage")
        interface.smooth('VEC2', "uv")
        info.vertex_out(interface)

        info.sampler(0, 'FLOAT_2D', "quiltTexture")
        info.fragment_out(0, 'VEC4', "fragColor")

        info.vertex_source("""
        void main() {
            uv = uv_in;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """)
        info.fragment_source("""
        void main() {
            fragColor = texture(quiltTexture, uv);
        }
        """)
        return gpu.shader.create_from_info(info)

    def draw_raw(self, texture: gpu.types.GPUTexture) -> None:
        self.blit_shader.bind()
        self.blit_shader.uniform_sampler("quiltTexture", texture)
        self.quad_batch.draw(self.blit_shader)

    def draw_interlaced(
        self, texture: gpu.types.GPUTexture, settings: LenticularSettings
    ) -> None:
        self.interlace_shader.bind()
        self.interlace_shader.uniform_sampler("quiltTexture", texture)
        self.interlace_shader.uniform_int("rows", settings.rows)
        self.interlace_shader.uniform_int("cols", settings.cols)
        self.interlace_shader.uniform_float("tilt", settings.tilt)
        self.interlace_shader.uniform_float("pitch", settings.pitch)
        self.interlace_shader.uniform_float("center", settings.center)
        self.interlace_shader.uniform_float("viewPortionElement", settings.view_portion)
        self.interlace_shader.uniform_float("subp", settings.subp)
        self.interlace_shader.uniform_float("focus", settings.focus)
        self.quad_batch.draw(self.interlace_shader)


# ==============================================================================
# 7. ON-SCREEN DISPLAY (OSD) SERVICE
# ==============================================================================

class DisplayOSD:
    """Renders HUD feedback directly over the secondary display canvas."""

    @staticmethod
    def draw(mode_name: str, focus_val: float) -> None:
        text = f"Mode: {mode_name} | Focus: {focus_val:+.4f} (Scroll / Drag / Arrows)"
        font_id = 0
        blf.size(font_id, 16)

        # Shadow drop
        blf.color(font_id, 0.0, 0.0, 0.0, 0.85)
        blf.position(font_id, 22, 23, 0)
        blf.draw(font_id, text)

        # Text foreground
        blf.color(font_id, 1.0, 1.0, 1.0, 0.95)
        blf.position(font_id, 20, 25, 0)
        blf.draw(font_id, text)


# ==============================================================================
# 8. OPERATORS & MODAL WINDOW CONTROLLER
# ==============================================================================

class OBJECT_OT_CreateLenticularRig(bpy.types.Operator):
    bl_idname = "object.create_lenticular_rig"
    bl_label = "Generate Camera Rig"
    bl_description = "Builds the parallel off-axis camera array"

    def execute(self, context: bpy.types.Context) -> set:
        views = CameraRigService.build_rig(context)
        self.report({'INFO'}, f"Built {views} off-axis camera perspectives.")
        return {'FINISHED'}


class RENDER_OT_BakeViewportQuilt(bpy.types.Operator):
    bl_idname = "render.bake_viewport_quilt"
    bl_label = "Capture Views to Quilt"
    bl_description = "Bakes the clean camera perspectives into the GPU Quilt Buffer"

    def execute(self, context: bpy.types.Context) -> set:
        try:
            views, width, height = QuiltBakerService.bake(context)
            self.report({'INFO'}, f"Baked {views} views to Quilt ({width}x{height}).")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class LENTICULAR_OT_ResetDefaults(bpy.types.Operator):
    bl_idname = "lenticular.reset_defaults"
    bl_label = "Reset to Defaults"
    bl_description = "Resets all parameters to optical defaults"

    def execute(self, context: bpy.types.Context) -> set:
        cfg = context.scene.lenticular_settings
        cfg.cols = LenticularDefaults.COLS
        cfg.rows = LenticularDefaults.ROWS
        cfg.tilt = LenticularDefaults.TILT
        cfg.pitch = LenticularDefaults.PITCH
        cfg.center = LenticularDefaults.CENTER
        cfg.subp = LenticularDefaults.SUBP
        cfg.view_portion = LenticularDefaults.VIEW_PORTION
        cfg.focus = LenticularDefaults.FOCUS
        cfg.baseline = LenticularDefaults.BASELINE
        cfg.focal_distance = LenticularDefaults.FOCAL_DISTANCE

        CameraRigService.build_rig(context)
        self.report({'INFO'}, "Lenticular settings reset to defaults.")
        return {'FINISHED'}


class WM_OT_LenticularRealtimeDisplay(bpy.types.Operator):
    bl_idname = "wm.lenticular_realtime_display"
    bl_label = "3D Display Window"

    _handle = None
    _timer = None
    _target_window: Optional[bpy.types.Window] = None
    _is_dragging: bool = False
    _last_mouse_x: int = 0
    _show_raw: bool = False

    def is_window_valid(self, context: bpy.types.Context) -> bool:
        return self._target_window is not None and self._target_window in list(
            context.window_manager.windows
        )

    def tag_redraw(self, context: bpy.types.Context) -> None:
        if self.is_window_valid(context):
            for area in self._target_window.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.tag_redraw()

    def step_focus(self, context: bpy.types.Context, delta: float) -> None:
        cfg = context.scene.lenticular_settings
        cfg.focus = max(-0.5, min(0.5, cfg.focus + delta))
        self.tag_redraw(context)

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        if not self.is_window_valid(context) or event.type in {'ESC', 'RIGHTMOUSE'}:
            self.cancel(context)
            return {'CANCELLED'}

        # Space: Toggle Raw / Interlaced
        if event.type == 'SPACE' and event.value == 'PRESS':
            self._show_raw = not self._show_raw
            self.tag_redraw(context)
            return {'RUNNING_MODAL'}

        step = 0.02 if not event.shift else 0.005

        # Scroll Wheel Focus Scrub
        if event.type == 'WHEELUP_MOUSE':
            self.step_focus(context, step)
            return {'RUNNING_MODAL'}
        elif event.type == 'WHEELDOWN_MOUSE':
            self.step_focus(context, -step)
            return {'RUNNING_MODAL'}

        # Arrow Keys Focus Scrub
        if event.type in {'UP_ARROW', 'RIGHT_ARROW'} and event.value in {'PRESS', 'REPEAT'}:
            self.step_focus(context, step)
            return {'RUNNING_MODAL'}
        elif event.type in {'DOWN_ARROW', 'LEFT_ARROW'} and event.value in {'PRESS', 'REPEAT'}:
            self.step_focus(context, -step)
            return {'RUNNING_MODAL'}

        # Left-Click Drag Scrub
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._is_dragging = True
                self._last_mouse_x = event.mouse_x
            elif event.value == 'RELEASE':
                self._is_dragging = False
            return {'RUNNING_MODAL'}

        elif event.type == 'MOUSEMOVE' and self._is_dragging:
            dx = event.mouse_x - self._last_mouse_x
            self._last_mouse_x = event.mouse_x
            factor = 0.002 if not event.shift else 0.0005
            self.step_focus(context, dx * factor)
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            self.tag_redraw(context)

        return {'PASS_THROUGH'}

    def draw_callback(self, context: bpy.types.Context) -> None:
        img = bpy.data.images.get(LenticularDefaults.BUFFER_NAME)
        if not img:
            return

        try:
            texture = gpu.texture.from_image(img)
        except Exception:
            return

        if not texture:
            return

        reg = context.region
        if reg:
            gpu.state.viewport_set(0, 0, reg.width, reg.height)

        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('NONE')

        pipeline = LenticularShaderPipeline.get()
        cfg = context.scene.lenticular_settings

        if self._show_raw:
            pipeline.draw_raw(texture)
            DisplayOSD.draw("RAW QUILT", cfg.focus)
        else:
            pipeline.draw_interlaced(texture, cfg)
            DisplayOSD.draw("3D INTERLACED", cfg.focus)

    def execute(self, context: bpy.types.Context) -> set:
        img = bpy.data.images.get(LenticularDefaults.BUFFER_NAME)
        if not img:
            bpy.ops.render.bake_viewport_quilt('EXEC_DEFAULT')

        bpy.ops.wm.window_new()
        self._target_window = context.window_manager.windows[-1]
        area = self._target_window.screen.areas[0]
        area.type = 'IMAGE_EDITOR'

        self._handle = area.spaces.active.draw_handler_add(
            self.draw_callback, (context,), 'WINDOW', 'POST_PIXEL'
        )
        self._timer = context.window_manager.event_timer_add(
            1.0 / 60.0, window=self._target_window
        )
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context: bpy.types.Context) -> None:
        if self._handle:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


# ==============================================================================
# 9. USER INTERFACE
# ==============================================================================

class VIEW3D_PT_LenticularControlPanel(bpy.types.Panel):
    bl_label = "3D Lenticular Display"
    bl_idname = "VIEW3D_PT_lenticular_control_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3D Lenticular"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        cfg = context.scene.lenticular_settings

        # Action Buttons
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator(
            "render.bake_viewport_quilt",
            icon='RENDER_STILL',
            text="1. Capture Views to Quilt",
        )
        col.operator(
            "wm.lenticular_realtime_display",
            icon='WINDOW',
            text="2. Open Display Window",
        )

        # Dynamic Focus
        box = layout.box()
        box.label(text="Dynamic Focus", icon='DRIVER_DISTANCE')
        box.prop(cfg, "focus", slider=True)
        box.label(text="In Window: Scroll / Drag / Arrows", icon='INFO')

        # Optical Calibration
        box = layout.box()
        box.label(text="Lenticular Optical Alignment", icon='MOD_MIRROR')
        col = box.column(align=True)
        col.prop(cfg, "pitch")
        col.prop(cfg, "tilt")
        col.prop(cfg, "center")
        col.prop(cfg, "subp")

        # Rig Configuration
        box = layout.box()
        box.label(text="Rig Array Setup", icon='CON_TRACKTO')
        row = box.row(align=True)
        row.prop(cfg, "cols")
        row.prop(cfg, "rows")
        col = box.column(align=True)
        col.prop(cfg, "baseline")
        col.prop(cfg, "focal_distance")

        layout.separator()
        layout.operator(
            "lenticular.reset_defaults", icon='LOOP_BACK', text="Reset to Defaults"
        )


# ==============================================================================
# 10. LIFECYCLE REGISTRATION
# ==============================================================================

CLASSES = (
    LenticularSettings,
    OBJECT_OT_CreateLenticularRig,
    RENDER_OT_BakeViewportQuilt,
    LENTICULAR_OT_ResetDefaults,
    WM_OT_LenticularRealtimeDisplay,
    VIEW3D_PT_LenticularControlPanel,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lenticular_settings = PointerProperty(type=LenticularSettings)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.lenticular_settings


if __name__ == "__main__":
    register()