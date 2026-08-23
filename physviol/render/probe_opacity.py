"""Can a body be faded out rather than switched off? Measured, not assumed.

Answer: **no**, not through this render path, and the segmentation pass makes it
moot even if it could.

Two mechanisms were tried against the pinned image (Kubric 2022.4.1 / Blender
2.93.4). Both accept keyframes -- the fcurves are created and visible on the
material's node tree -- and neither moves the rendered pixels:

    material.transmission 0 -> 1   ball mean RGB 153 -> 150   (over five frames)
    Principled BSDF Alpha 1 -> 0   ball mean RGB 131 -> 134

The second is the decisive one, and not because of the pixels: the segmentation
pass reported the same 3976 object pixels at alpha 0 as at alpha 1. Cryptomatte
tracks geometry, so a body that faded to nothing visually would still be fully
present in `seg.npz` -- and every mask, every observability window and every
residual in this project is computed from that. A fade the annotation cannot see
is worse than no fade: the clip would show a violation and the labels would deny
it.

So `permanence` produces its gradient geometrically, by shrinking the body away
over several frames before removing it. That reads as a dissolve, and the
segmentation shrinks with it, so the mask and the observability clock follow the
violation instead of lagging behind it.

Kept for the same reason as `probe_fluid.py`: so the next person to want a fade
finds the measurement rather than repeating it.

    bash docker/kubric.sh physviol/render/probe_opacity.py
"""
import sys, os
sys.path.insert(0, "/kubric")
import numpy as np, kubric as kb
from kubric.renderer import Blender

scratch = "/kubric/out/_probe"; os.makedirs(scratch, exist_ok=True)
scene = kb.Scene(resolution=(96, 96), frame_start=0, frame_end=4, frame_rate=12)
r = Blender(scene, scratch, samples_per_pixel=16, background_transparency=False)
scene.background = kb.Color(0.05, 0.05, 0.06)
mat = kb.PrincipledBSDFMaterial(color=kb.Color(0.85, 0.15, 0.15), roughness=0.5)
ball = kb.Sphere(name="ball", scale=(0.6,)*3, position=(0, 0, 0.6),
                 static=True, material=mat, segmentation_id=2)
scene += ball
scene += kb.Cube(name="floor", scale=(4, 4, 0.1), position=(0, 0, -0.1),
                 static=True, segmentation_id=1,
                 material=kb.PrincipledBSDFMaterial(color=kb.Color(.35,.35,.4)))
scene += kb.DirectionalLight(name="sun", position=(-2,-2,4), look_at=(0,0,0), intensity=2.2)
scene += kb.PerspectiveCamera(name="cam", position=(0,-4,1.2), look_at=(0,0,0.6))

bmat = ball.material.linked_objects[r]
alpha = bmat.node_tree.nodes["Principled BSDF"].inputs["Alpha"]
for f, a in ((0, 1.0), (2, 0.5), (4, 0.0)):
    alpha.default_value = a
    alpha.keyframe_insert("default_value", frame=f)
print("ALPHA fcurves:",
      [(fc.data_path, len(fc.keyframe_points))
       for fc in bmat.node_tree.animation_data.action.fcurves
       if "Alpha" in fc.data_path or "inputs[19]" in fc.data_path or True][:6])

out = r.render(return_layers=["rgba", "segmentation"])
rgba, seg = np.asarray(out["rgba"]), np.asarray(out["segmentation"])
for f in range(rgba.shape[0]):
    m = seg[f, ..., 0] == 2
    print("frame %d  segpx %4d  ball mean rgb %s"
          % (f, int(m.sum()),
             np.round(rgba[f][m][:, :3].mean(0), 1) if m.any() else "-"))
